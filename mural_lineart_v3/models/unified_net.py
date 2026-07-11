"""
Y型统一生成器：共享编码器 + 双解码器
  分支 A: 纯提取 → 残缺粗线稿
  分支 B: ResBlock修复 → 完整线稿
"""

import torch
import torch.nn as nn
from .inception import InceptionBlock
from .attention import AttentionGate
from .modules import ResnetBlock, SelfAttention, GatedConv2d


def _make_decoder_layers(num_downs, enc_channels, ch, dec_ch):
    """构建一组解码器层（Up + Attention + Inception + out_conv）"""
    up_blocks = nn.ModuleList()
    dec_inceptions = nn.ModuleList()
    attentions = nn.ModuleList()

    for i in range(num_downs):
        prev_dec_ch = dec_ch
        dec_ch = dec_ch // 2
        skip_ch = enc_channels[-(i + 2)]
        attentions.append(AttentionGate(dec_ch, skip_ch, dec_ch // 2))
        up_blocks.append(nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(prev_dec_ch, dec_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dec_ch),
            nn.ReLU(inplace=True),
        ))
        dec_inceptions.append(InceptionBlock(dec_ch + skip_ch, dec_ch))

    out_conv = nn.Sequential(
        nn.Conv2d(dec_ch, 1, kernel_size=3, padding=1),
        nn.Sigmoid(),
    )
    return up_blocks, dec_inceptions, attentions, out_conv


class UnifiedGenerator(nn.Module):
    """共享编码器 + 双解码器"""

    def __init__(self, in_ch=4, base_ch=64, num_downs=5, num_res=8):
        super().__init__()
        self.num_downs = num_downs

        # ── 共享编码器 ──
        # 门控守门员：在入口处过滤 Mask 边界伪影
        self.gated_input = GatedConv2d(in_ch, base_ch, kernel_size=3, padding=1)
        self.enc_in = InceptionBlock(base_ch, base_ch)
        self.enc_inceptions = nn.ModuleList()
        self.enc_downs = nn.ModuleList()

        ch = base_ch
        self._enc_channels = [base_ch]

        for _ in range(num_downs):
            next_ch = min(ch * 2, 512)
            self.enc_inceptions.append(InceptionBlock(ch, ch))
            self.enc_downs.append(nn.Sequential(
                nn.Conv2d(ch, next_ch, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(next_ch), nn.ReLU(inplace=True),
            ))
            ch = next_ch
            self._enc_channels.append(ch)

        # 轻量Mask Encoder：将0/1 mask转为8ch丰富特征（比原始mask信息量大得多）
        self.mask_encoders = nn.ModuleList()
        for _ in range(num_downs):
            self.mask_encoders.append(nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.InstanceNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 8, kernel_size=3, padding=1),
                nn.InstanceNorm2d(8), nn.ReLU(inplace=True),
            ))

        # 多尺度Mask融合：前3层用GatedConv防伪影，后2层用1×1 Conv
        ch_fusion = base_ch
        self.mask_fusions = nn.ModuleList()
        for i in range(num_downs):
            if i < 3:
                self.mask_fusions.append(
                    GatedConv2d(ch_fusion + 8, ch_fusion, kernel_size=3, padding=1))
            else:
                self.mask_fusions.append(
                    nn.Conv2d(ch_fusion + 8, ch_fusion, kernel_size=1))
            ch_fusion = min(ch_fusion * 2, 512)

        # ── 分支 A 瓶颈（轻量修复）──
        self.bottleneck_A = nn.Sequential(
            ResnetBlock(ch), ResnetBlock(ch), ResnetBlock(ch),
        )

        # ── 分支 B 瓶颈（ResBlock + SelfAttention 修复）──
        res_blocks = [
            ResnetBlock(ch), ResnetBlock(ch), ResnetBlock(ch),
            SelfAttention(ch),
            ResnetBlock(ch), ResnetBlock(ch), ResnetBlock(ch),
        ]
        self.bottleneck_B = nn.Sequential(*res_blocks)

        # ── 双解码器 ──
        self.dec_A_up, self.dec_A_inv, self.dec_A_attn, self.out_A = \
            _make_decoder_layers(num_downs, self._enc_channels, ch, ch)
        self.dec_B_up, self.dec_B_inv, self.dec_B_attn, self.out_B = \
            _make_decoder_layers(num_downs, self._enc_channels, ch, ch)

    def _decode(self, feat, up_blocks, dec_inceptions, attentions, out_conv, skips,
                return_features=False):
        feats = []
        for i in range(self.num_downs):
            feat = up_blocks[i](feat)
            skip = skips[-(i + 2)]
            attn_skip, _ = attentions[i](feat, skip)
            attn_skip = skip + attn_skip
            if attn_skip.shape[2:] != feat.shape[2:]:
                padY = attn_skip.shape[2] - feat.shape[2]
                padX = attn_skip.shape[3] - feat.shape[3]
                feat = nn.functional.pad(feat, [padX // 2, padX - padX // 2,
                                                 padY // 2, padY - padY // 2])
            feat_cat = torch.cat([feat, attn_skip], dim=1)
            feat = dec_inceptions[i](feat_cat) + feat
            if return_features:
                feats.append(feat)
        out = out_conv(feat)
        return (out, feats) if return_features else out

    def forward(self, img, mask, return_decoder_features=False):
        x = torch.cat([img, mask], dim=1)  # [B, 4, H, W]

        # 共享编码（多尺度Mask Encoder注入）
        skips = []
        x_gated = self.gated_input(x)
        feat = self.enc_in(x_gated)
        skips.append(feat)
        for i, (inception, down, mask_enc, mask_fusion) in enumerate(zip(
            self.enc_inceptions, self.enc_downs,
            self.mask_encoders, self.mask_fusions)):
            feat = inception(feat)
            mask_down = nn.functional.interpolate(
                mask, size=feat.shape[2:], mode='nearest')
            mask_feat = mask_enc(mask_down)  # 1ch → 8ch 丰富特征
            feat = torch.cat([feat, mask_feat], dim=1)
            feat = mask_fusion(feat)
            feat = down(feat)
            skips.append(feat)

        # 分支 A
        feat_A = self.bottleneck_A(feat)
        rA = self._decode(feat_A, self.dec_A_up, self.dec_A_inv,
                          self.dec_A_attn, self.out_A, skips,
                          return_features=return_decoder_features)

        # 分支 B
        feat_B = self.bottleneck_B(feat)
        rB = self._decode(feat_B, self.dec_B_up, self.dec_B_inv,
                          self.dec_B_attn, self.out_B, skips,
                          return_features=return_decoder_features)

        if return_decoder_features:
            out_A, feats_A = rA
            out_B, feats_B = rB
            return out_A, out_B, feats_A, feats_B
        return rA, rB
