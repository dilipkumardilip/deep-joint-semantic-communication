"""
Perceptual and Structural Loss Functions for High-Definition Deep JSCC.

Combines:
- Mean Squared Error (MSE / L2) for energy conservation & PSNR
- Mean Absolute Error (L1) for sharper gradient edges
- Structural Similarity Index (SSIM) for human visual perceptual realism
"""

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.tensor(
        [
            math.exp(-((x - window_size // 2) ** 2) / (2.0 * sigma ** 2))
            for x in range(window_size)
        ],
        dtype=torch.float32,
    )
    return gauss / gauss.sum()


def _create_window(window_size: int, channels: int) -> torch.Tensor:
    _1d = _gaussian_window(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
    val_range: float = 1.0,
) -> torch.Tensor:
    """
    Computes Structural Similarity Index (SSIM) between two batches of images.

    Args:
        img1 (torch.Tensor): Reconstructed images (B, C, H, W) in [0, 1].
        img2 (torch.Tensor): Reference images (B, C, H, W) in [0, 1].
        window_size (int): Size of Gaussian filter window (default: 11).
        size_average (bool): Average over spatial and batch dimensions.
        val_range (float): Dynamic range of pixel values (1.0 for normalized images).

    Returns:
        torch.Tensor: SSIM value (scalar if size_average=True).
    """
    channels = img1.size(1)
    window = _create_window(window_size, channels).to(dtype=img1.dtype, device=img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    c1 = (0.01 * val_range) ** 2
    c2 = (0.03 * val_range) ** 2

    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=[1, 2, 3])


class SSIMLoss(nn.Module):
    """Structural Dissimilarity (1 - SSIM) Loss."""

    def __init__(self, window_size: int = 11, val_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.val_range = val_range

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim(img1, img2, window_size=self.window_size, val_range=self.val_range)


class CompositeJSCCLoss(nn.Module):
    """
    Multi-objective Loss for Deep JSCC Image Transmission:
      Loss = alpha * MSE + beta * L1 + gamma * (1 - SSIM)

    Default weights:
      - alpha = 0.5 (MSE preserves energy / PSNR)
      - beta  = 0.3 (L1 produces sharper edges without blur)
      - gamma = 0.2 (SSIM preserves structural textures & visual realism)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        window_size: int = 11,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.ssim_loss = SSIMLoss(window_size=window_size)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        loss_mse = self.mse(pred, target)
        loss_l1 = self.l1(pred, target)
        loss_ssim = self.ssim_loss(pred, target)

        total_loss = (
            self.alpha * loss_mse
            + self.beta * loss_l1
            + self.gamma * loss_ssim
        )

        metrics = {
            "total_loss": total_loss.item(),
            "mse": loss_mse.item(),
            "l1": loss_l1.item(),
            "ssim": (1.0 - loss_ssim).item(),
        }
        return total_loss, metrics
