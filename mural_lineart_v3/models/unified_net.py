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
            nn.BatchNorm2d(dec_ch),
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
                nn.BatchNorm2d(next_ch), nn.ReLU(inplace=True),
            ))
            ch = next_ch
            self._enc_channels.append(ch)

        # ── 分支 A 瓶颈（纯提取）──
        self.bottleneck_A = nn.Sequential(
            InceptionBlock(ch, ch),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
        )

        # ── 分支 B 瓶颈（ResBlock + SelfAttention 修复）──
        res_blocks = [
            ResnetBlock(ch), ResnetBlock(ch), ResnetBlock(ch),
            SelfAttention(ch),  # 全局视野，跨越大空洞连接线条
            ResnetBlock(ch), ResnetBlock(ch), ResnetBlock(ch),
        ]
        self.bottleneck_B = nn.Sequential(*res_blocks)

        # ── 双解码器 ──
        self.dec_A_up, self.dec_A_inv, self.dec_A_attn, self.out_A = \
            _make_decoder_layers(num_downs, self._enc_channels, ch, ch)
        self.dec_B_up, self.dec_B_inv, self.dec_B_attn, self.out_B = \
            _make_decoder_layers(num_downs, self._enc_channels, ch, ch)

    def _decode(self, feat, up_blocks, dec_inceptions, attentions, out_conv, skips):
        for i in range(self.num_downs):
            feat = up_blocks[i](feat)
            skip = skips[-(i + 2)]
            attn_skip, _ = attentions[i](feat, skip)
            if attn_skip.shape[2:] != feat.shape[2:]:
                diffY = attn_skip.shape[2] - feat.shape[2]
                diffX = attn_skip.shape[3] - feat.shape[3]
                feat = nn.functional.pad(feat, [diffX // 2, diffX - diffX // 2,
                                                 diffY // 2, diffY - diffY // 2])
            feat = torch.cat([feat, attn_skip], dim=1)
            feat = dec_inceptions[i](feat)
        return out_conv(feat)

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)  # [B, 4, H, W]

        # 共享编码
        skips = []
        # 门控过滤：先扼杀 Mask 边界假边缘
        x_gated = self.gated_input(x)
        feat = self.enc_in(x_gated)
        skips.append(feat)
        for inception, down in zip(self.enc_inceptions, self.enc_downs):
            feat = inception(feat)
            feat = down(feat)
            skips.append(feat)

        # 分支 A
        feat_A = self.bottleneck_A(feat)
        out_A = self._decode(feat_A, self.dec_A_up, self.dec_A_inv,
                             self.dec_A_attn, self.out_A, skips)

        # 分支 B
        feat_B = self.bottleneck_B(feat)
        out_B = self._decode(feat_B, self.dec_B_up, self.dec_B_inv,
                             self.dec_B_attn, self.out_B, skips)

        return out_A, out_B
