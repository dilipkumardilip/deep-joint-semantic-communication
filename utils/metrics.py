"""
Numerical and Signal Processing Metrics for Deep JSCC.
"""

import math
import torch
import torch.nn as nn


def calculate_psnr(mse: float, max_val: float = 1.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Formula: PSNR = 10 * log10(max_val^2 / MSE)

    Args:
        mse (float): Mean Squared Error between original and reconstructed images.
        max_val (float): Maximum possible pixel value (1.0 for normalized tensors).

    Returns:
        float: PSNR in dB (capped at 100.0 dB for numerical stability as MSE -> 0).
    """
    if mse <= 1e-10:
        return 100.0
    return 10.0 * math.log10((max_val ** 2) / mse)


def calculate_mse(
    img1: torch.Tensor,
    img2: torch.Tensor,
) -> float:
    """
    Computes Mean Squared Error between two image tensors.

    Args:
        img1 (torch.Tensor): First image tensor.
        img2 (torch.Tensor): Second image tensor.

    Returns:
        float: Scalar MSE value.
    """
    criterion = nn.MSELoss()
    return float(criterion(img1, img2).item())
