"""
Pixel-wise Absolute Reconstruction Error Heatmap Visualizations for Deep JSCC.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from .style import set_plot_style


def plot_error_heatmaps(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    snr: float = 10.0,
    num_samples: int = 4,
    save_path: str = os.path.join(config.OUTPUTS_DIR, "error_heatmaps.png"),
) -> None:
    """
    Visualizes original images, reconstructed images, and their absolute pixel error heatmaps.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()

    images, _ = next(iter(loader))
    images = images[:num_samples].to(device)

    with torch.no_grad():
        reconstructed = model(images, snr_db=snr).clamp(0.0, 1.0)

    diff = torch.abs(images - reconstructed).mean(dim=1).cpu().numpy()  # Average across RGB
    images = images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()

    fig, axes = plt.subplots(num_samples, 3, figsize=(8.5, num_samples * 2.5))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(num_samples):
        # 1. Original
        axes[i, 0].imshow(np.transpose(images[i], (1, 2, 0)))
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("Original", fontweight="bold")

        # 2. Reconstructed
        axes[i, 1].imshow(np.transpose(reconstructed[i], (1, 2, 0)))
        axes[i, 1].axis("off")
        if i == 0:
            axes[i, 1].set_title(f"Reconstructed ({snr} dB)", fontweight="bold")

        # 3. Error Heatmap
        heatmap = axes[i, 2].imshow(diff[i], cmap="inferno", vmin=0.0, vmax=0.3)
        axes[i, 2].axis("off")
        if i == 0:
            axes[i, 2].set_title("Error Heatmap |x - x̂|", fontweight="bold")
        fig.colorbar(heatmap, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"-> Saved error heatmaps: {save_path}")
