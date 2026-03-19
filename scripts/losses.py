"""
scripts/losses.py — Loss functions for Phase 1 and Phase 2.

Phase 1: L1Loss (plain wrapper)
Phase 2: CombinedLoss(alpha=0.7) = alpha * L1 + (1 - alpha) * (1 - SSIM)

SSIM is differentiable via F.conv2d with a Gaussian window of size 11.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create a 2D Gaussian kernel as a (1, 1, size, size) tensor."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.unsqueeze(0) * g.unsqueeze(1)   # (size, size)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.unsqueeze(0).unsqueeze(0)      # (1, 1, size, size)


def _ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window: torch.Tensor,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    Differentiable SSIM between pred and target, both (B, 1, H, W) in [0, 1].
    Returns scalar mean SSIM over the batch and spatial positions.
    """
    padding = window.shape[-1] // 2

    mu_x = F.conv2d(pred, window, padding=padding)
    mu_y = F.conv2d(target, window, padding=padding)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy   = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred * pred,     window, padding=padding) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, window, padding=padding) - mu_y_sq
    sigma_xy   = F.conv2d(pred * target,   window, padding=padding) - mu_xy

    numerator   = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    return (numerator / denominator).mean()


class L1Loss(nn.Module):
    """Plain L1 loss — used in Phase 1."""

    def __init__(self) -> None:
        super().__init__()
        self._l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._l1(pred, target)


class CombinedLoss(nn.Module):
    """
    CombinedLoss for Phase 2:
        loss = alpha * L1 + (1 - alpha) * (1 - SSIM)

    Default alpha = 0.7.
    Gaussian window (size=11) is registered as a buffer so it moves with
    the model to the correct device automatically.
    """

    def __init__(self, alpha: float = 0.7, window_size: int = 11) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self._l1 = nn.L1Loss()
        window = _gaussian_kernel(window_size)
        self.register_buffer("window", window)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = self._l1(pred, target)
        ssim_val = _ssim(pred, target, self.window)
        return self.alpha * l1 + (1.0 - self.alpha) * (1.0 - ssim_val)
