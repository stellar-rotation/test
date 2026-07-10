"""
阶段一 Pix2Pix Generator (单路) + PatchGAN Discriminator
U-Net + InceptionBlock + AttentionGate
"""

import torch
import torch.nn as nn
from .inception import InceptionBlock
from .attention import AttentionGate


class ExtractionGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base_ch=64, num_downs=5):
        super().__init__()
        self.num_downs = num_downs

        self.enc_in = InceptionBlock(in_ch, base_ch)
        self.enc_inceptions = nn.ModuleList()
        self.enc_downs = nn.ModuleList()

        ch = base_ch
        self._enc_channels = [base_ch]

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

        self.bottleneck = nn.Sequential(
            InceptionBlock(ch, ch),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

        self.attentions = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.dec_inceptions = nn.ModuleList()
        dec_ch = ch
        for i in range(num_downs):
            prev_dec_ch = dec_ch
            dec_ch = dec_ch // 2
            skip_ch = self._enc_channels[-(i + 2)]
            self.attentions.append(AttentionGate(dec_ch, skip_ch, dec_ch // 2))
            self.up_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(prev_dec_ch, dec_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(dec_ch),
                    nn.ReLU(inplace=True),
                )
            )
            self.dec_inceptions.append(InceptionBlock(dec_ch + skip_ch, dec_ch))

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


class PatchGANDiscriminator(nn.Module):
    """保留以备后续使用"""

    def __init__(self, in_ch=4, base_ch=64, n_layers=3):
        super().__init__()
        from torch.nn.utils import spectral_norm
        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, base_ch, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ))
        ch = base_ch
        for _ in range(1, n_layers):
            next_ch = min(ch * 2, 512)
            self.blocks.append(nn.Sequential(
                spectral_norm(nn.Conv2d(ch, next_ch, kernel_size=4, stride=2, padding=1)),
                nn.BatchNorm2d(next_ch), nn.LeakyReLU(0.2, inplace=True),
            ))
            ch = next_ch
        next_ch = min(ch * 2, 512)
        self.blocks.append(nn.Sequential(
            spectral_norm(nn.Conv2d(ch, next_ch, kernel_size=4, stride=1, padding=1)),
            nn.BatchNorm2d(next_ch), nn.LeakyReLU(0.2, inplace=True),
        ))
        self.blocks.append(nn.Sequential(
            spectral_norm(nn.Conv2d(next_ch, 1, kernel_size=4, stride=1, padding=1)),
        ))

    def forward(self, x, return_features=False):
        feats = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if return_features and i < len(self.blocks) - 1:
                feats.append(x)
        return (x, feats) if return_features else x
