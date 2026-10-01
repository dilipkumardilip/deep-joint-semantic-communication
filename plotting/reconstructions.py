"""
Reconstruction Visualization Grids and Image Comparisons for Deep JSCC.
"""

import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from .style import set_plot_style


def plot_reconstruction_grid(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    snr_list: List[float],
    num_samples: int = 5,
    save_path: str = os.path.join(config.OUTPUTS_DIR, "reconstruction_grid.png"),
) -> None:
    """
    Plots a grid comparing original CIFAR-10 images with their reconstructions across various SNRs.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()

    images, _ = next(iter(loader))
    images = images[:num_samples].to(device)

    reconstructions = {}
    with torch.no_grad():
        for snr in snr_list:
            reconstructions[snr] = model(images, snr_db=snr).cpu().clamp(0.0, 1.0)

    images = images.cpu()
    rows = num_samples
    cols = 1 + len(snr_list)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(rows):
        orig_img = images[i].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(orig_img)
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("Original", fontweight="bold", fontsize=12)

        for j, snr in enumerate(snr_list):
            recon_img = reconstructions[snr][i].permute(1, 2, 0).numpy()
            axes[i, j + 1].imshow(recon_img)
            axes[i, j + 1].axis("off")
            if i == 0:
                axes[i, j + 1].set_title(f"SNR {snr} dB", fontweight="bold", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"-> Saved reconstruction grid: {save_path}")


def plot_reconstruction_comparison(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    snr_list: List[float],
    num_samples: int = 5,
    save_path: str = os.path.join(config.OUTPUTS_DIR, "reconstruction_comparison.png"),
) -> None:
    """
    Saves a side-by-side visual comparison of original vs reconstructed images
    transmitted over channels with different SNR values.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()

    images, _ = next(iter(loader))
    images = images[:num_samples].to(device)

    reconstructions = {}
    with torch.no_grad():
        for snr in snr_list:
            reconstructions[snr] = model(images, snr_db=snr).cpu().clamp(0.0, 1.0)

    images = images.cpu()
    cols = 1 + len(snr_list)
    rows = num_samples
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))

    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(rows):
        orig_img = images[i].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(orig_img)
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("Original", fontsize=12, fontweight="bold")

        for j, snr in enumerate(snr_list):
            recon_img = reconstructions[snr][i].permute(1, 2, 0).numpy()
            axes[i, j + 1].imshow(recon_img)
            axes[i, j + 1].axis("off")
            if i == 0:
                axes[i, j + 1].set_title(f"SNR = {snr} dB", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"\nVisual comparison saved to: {save_path}")
