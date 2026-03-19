"""
models/rcan.py — Residual Channel Attention Network (RCAN) for 4x SAR SR.

REDUCED config for hardware-constrained training (single RTX 5060 Laptop, 8 GB):
  num_features=64, num_groups=5, num_rcab=10, scale=4  (~3.2M params)
  (Paper used 10 groups x 20 RCABs on 4x A6000 GPUs)

Architecture (Zhang et al., ECCV 2018):
  HeadConv
    -> 5 x ResidualGroup (each: 10 x RCAB + Conv3x3, long skip)
  -> Conv3x3 + global skip to head features
  -> PixelShuffle x4 upsampler
  -> TailConv
  -> clamp(0, 1)

Key constraints:
  - NO BatchNorm anywhere
  - PixelShuffle upscaling
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, n_feats: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(n_feats, n_feats // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(n_feats // reduction, n_feats, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block."""

    def __init__(self, n_feats: int, reduction: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        self.ca = ChannelAttention(n_feats, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ca(self.body(x)) + x


class ResidualGroup(nn.Module):
    """Stack of RCABs with a long residual skip."""

    def __init__(
        self, n_feats: int, n_resblocks: int = 10, reduction: int = 16
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            *[RCAB(n_feats, reduction) for _ in range(n_resblocks)],
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + x


class RCAN(nn.Module):
    """
    Reduced RCAN: 5 ResidualGroups x 10 RCABs, PixelShuffle x4, ~3.2M params.
    Input:  (B, 1, 64, 64)   float32 [0, 1]
    Output: (B, 1, 256, 256) float32 [0, 1]
    """

    def __init__(
        self,
        n_feats:    int = 64,
        n_resgroups: int = 5,
        n_resblocks: int = 10,
        reduction:  int = 16,
        scale:      int = 4,
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(1, n_feats, 3, padding=1)

        self.body = nn.Sequential(
            *[ResidualGroup(n_feats, n_resblocks, reduction) for _ in range(n_resgroups)]
        )
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, padding=1)

        # PixelShuffle x4: need n_feats * scale^2 channels before shuffle
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
        )
        self.tail = nn.Conv2d(n_feats, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_feat = self.head(x)                         # (B, C, 64, 64)
        feat = self.body(head_feat)
        feat = self.body_tail(feat) + head_feat          # global skip
        feat = self.upsample(feat)                       # (B, C, 256, 256)
        out  = self.tail(feat)                           # (B, 1, 256, 256)
        return out.clamp(0.0, 1.0)
