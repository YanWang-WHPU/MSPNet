from __future__ import annotations

import torch
from torch import nn


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(ConvBNAct(in_ch, out_ch), ConvBNAct(out_ch, out_ch))


class MSPNet(nn.Module):
    """HSI-only deployment with PASP buffers and an optional CABG boundary head."""
    def __init__(self, in_ch: int = 32, num_classes: int = 3, base: int = 24, boundary_head: bool = False,
                 spectral_prototypes: torch.Tensor | None = None,
                 channel_mean: torch.Tensor | None = None, channel_std: torch.Tensor | None = None):
        super().__init__()
        if spectral_prototypes is not None:
            if channel_mean is None or channel_std is None:
                raise ValueError("Prototype input requires channel normalization statistics.")
            self.register_buffer("spectral_prototypes", spectral_prototypes.clone())
            self.register_buffer("channel_mean", channel_mean.clone())
            self.register_buffer("channel_std", channel_std.clone())
            in_ch += spectral_prototypes.shape[0]
        else:
            self.spectral_prototypes = None
            self.channel_mean = None
            self.channel_std = None
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 12)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(base * 12, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, num_classes, 1)
        self.boundary_head = nn.Conv2d(base, 1, 1) if boundary_head else None

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.spectral_prototypes is not None:
            raw = x * self.channel_std[None, :, None, None] + self.channel_mean[None, :, None, None]
            raw = raw - raw.mean(dim=1, keepdim=True)
            raw = raw / (raw.square().sum(dim=1, keepdim=True).sqrt() + 1e-6)
            similarity = torch.einsum("bchw,kc->bkhw", raw, self.spectral_prototypes)
            x = torch.cat([x, similarity], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.head(d1)
        if self.boundary_head is not None:
            return logits, self.boundary_head(d1)
        return logits



# Keep the historical name for existing training scripts and checkpoints.
HSIUNet = MSPNet
