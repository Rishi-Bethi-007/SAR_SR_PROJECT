"""
models/srcnn.py — SRCNN baseline (Phase 1).

Architecture (Dong et al. classic, ~57K params):
  1. Bicubic upsample input (1, 64, 64) -> (1, 256, 256)
  2. Conv2d(1,  64, 9x9, padding=4) -> ReLU
  3. Conv2d(64, 32, 5x5, padding=2) -> ReLU
  4. Conv2d(32,  1, 5x5, padding=2)
  5. Global residual: add bicubic upsampled input
  6. Clamp output to [0, 1]

Note: the middle conv uses 5x5 (not 1x1) to reach the ~57K param target.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        # Zero-init the reconstruction layer so training starts at bicubic
        # performance (~27 dB) rather than fighting random-weight noise (~20 dB).
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 64, 64)
        bicubic = F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False)
        # bicubic: (B, 1, 256, 256)

        feat = F.relu(self.conv1(bicubic), inplace=True)
        feat = F.relu(self.conv2(feat),    inplace=True)
        feat = self.conv3(feat)

        # Global residual: learned correction on top of bicubic
        out = feat + bicubic
        return out.clamp(0.0, 1.0)
