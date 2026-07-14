"""
阶段一：线稿提取模型
Pix2Pix 框架 — U-Net(Inception+Attention) 生成器 + PatchGAN 判别器
"""

import torch
import torch.nn as nn
from .inception import InceptionBlock
from .attention import AttentionGate

class DilatedResBlock(nn.Module):
    """带空洞卷积的残差块，用于指数级扩大感受野"""
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + res)

class ExtractionGenerator(nn.Module):
    """
    U-Net 编码器-解码器
    - 编码器每层: InceptionBlock → DownConv(stride=2)
    - 解码器每层: Upsample → AttentionGate(decoder_feat, skip_feat) → InceptionBlock
    """

    def __init__(self, in_ch=4, out_ch=1, base_ch=64, num_downs=5):
        super().__init__()
        self.num_downs = num_downs

        # ── 编码器：先建好，同时记录每层输出通道 ──
        self.enc_in = InceptionBlock(in_ch, base_ch)

        self.enc_inceptions = nn.ModuleList()
        self.enc_downs = nn.ModuleList()

        ch = base_ch
        self._enc_channels = [base_ch]  # 记录每层输出通道数（含 enc_in）

        for _ in range(num_downs):
            next_ch = min(ch * 2, 512)
            self.enc_inceptions.append(InceptionBlock(ch, ch))
            self.enc_downs.append(
                nn.Sequential(
                    nn.Conv2d(ch, next_ch, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(next_ch),
                    nn.ReLU(inplace=True),
                )
            )
            ch = next_ch
            self._enc_channels.append(ch)

        # ch 现在是 bottleneck 通道数
        # 引入级联的空洞残差块 (膨胀率 1, 2, 4 适配 16x16 的特征图分辨率)
        self.bottleneck = nn.Sequential(
            InceptionBlock(ch, ch),
            DilatedResBlock(ch, dilation=1),
            DilatedResBlock(ch, dilation=2),
            DilatedResBlock(ch, dilation=4),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

        # ── 解码器：从 _enc_channels 反推 skip 通道 ──
        # enc_channels = [64, 128, 256, 512, 512, 512] (base_ch=64, num_downs=5 时)
        # bottleneck = 512 (最深层)
        # 解码器第 i 层对应 skip = enc_channels[-(i+2)]
        self.attentions = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.dec_inceptions = nn.ModuleList()

        dec_ch = ch  # bottleneck channels
        for i in range(num_downs):
            prev_dec_ch = dec_ch
            dec_ch = dec_ch // 2
            skip_ch = self._enc_channels[-(i + 2)]  # 对应层的编码器输出通道

            self.attentions.append(AttentionGate(dec_ch, skip_ch, dec_ch // 2))
            self.up_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(prev_dec_ch, dec_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(dec_ch),
                    nn.ReLU(inplace=True),
                )
            )
            # concat: dec_feat(ch) + attn_skip(skip_ch) → dec_ch + skip_ch
            self.dec_inceptions.append(InceptionBlock(dec_ch + skip_ch, dec_ch))

        # ── 输出头 ──
        self.out_conv = nn.Sequential(
            nn.Conv2d(dec_ch, out_ch, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        skips = []
        feat = self.enc_in(x)
        skips.append(feat)

        for inception, down in zip(self.enc_inceptions, self.enc_downs):
            feat = inception(feat)
            feat = down(feat)
            skips.append(feat)

        feat = self.bottleneck(feat)

        for i in range(self.num_downs):
            feat = self.up_blocks[i](feat)
            skip = skips[-(i + 2)]
            attn_skip, _ = self.attentions[i](feat, skip)
            feat = torch.cat([feat, attn_skip], dim=1)
            feat = self.dec_inceptions[i](feat)

        return self.out_conv(feat)


# ──────────────────────────────────────────────────────────────────
# PatchGAN 判别器
# ──────────────────────────────────────────────────────────────────

class PatchGANDiscriminator(nn.Module):
    """70×70 PatchGAN + Spectral Norm（防梯度爆炸）"""

    def __init__(self, in_ch=4, base_ch=64, n_layers=3):
        super().__init__()
        from torch.nn.utils import spectral_norm

        sequence = [
            spectral_norm(nn.Conv2d(in_ch, base_ch, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_ch
        for _ in range(1, n_layers):
            next_ch = min(ch * 2, 512)
            sequence += [
                spectral_norm(nn.Conv2d(ch, next_ch, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = next_ch
        next_ch = min(ch * 2, 512)
        sequence += [
            spectral_norm(nn.Conv2d(ch, next_ch, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        sequence += [spectral_norm(nn.Conv2d(next_ch, 1, kernel_size=4, stride=1, padding=1))]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
