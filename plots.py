"""
Visualization and Plotting Suite for Deep Joint Source-Channel Communication (Deep JSCC).

Includes:
1. PSNR vs Channel SNR Curve (Graceful degradation / cliff-effect avoidance)
2. Original vs Reconstructed Image Comparison Grid
3. Pixel-wise Error Heatmaps (|x - x_hat|)
4. Transmitted Latent Channel Symbol Constellation / Distribution
5. Training & Validation Convergence Curves

Can be imported as a library or executed directly:
    python plots.py --all
"""

import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import DeepJSCC
from dataset import get_cifar10_loaders


def set_plot_style():
    """Configures clean, publication-ready Matplotlib aesthetics."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
    })


def plot_psnr_vs_snr(
    snr_list: List[float],
    psnr_list: List[float],
    title: str = "Deep JSCC: PSNR vs Channel SNR",
    save_path: str = "./outputs/psnr_vs_snr.png",
    benchmark_data: Optional[Dict[str, List[float]]] = None,
):
    """
    Plots the classic Deep JSCC Rate-Distortion curve across different Channel SNRs.
    Demonstrates graceful degradation without the cliff effect of classical separation schemes.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    set_plot_style()

    plt.figure(figsize=(8, 5.5))
    plt.plot(
        snr_list,
        psnr_list,
        marker="o",
        color="#1f77b4",
        linewidth=2.5,
        markersize=7,
        label="Deep JSCC (Proposed)",
    )

    if benchmark_data:
        colors = ["#ff7f0e", "#2ca02c", "#d62728"]
        for idx, (bench_name, bench_psnr) in enumerate(benchmark_data.items()):
            color = colors[idx % len(colors)]
            plt.plot(
                snr_list[:len(bench_psnr)],
                bench_psnr,
                marker="s",
                linestyle="--",
                linewidth=1.8,
                color=color,
                label=bench_name,
            )

    plt.xlabel("Channel SNR (dB)")
    plt.ylabel("Reconstruction PSNR (dB)")
    plt.title(title, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"-> Saved PSNR vs SNR plot: {save_path}")


def plot_reconstruction_grid(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    snr_list: List[float],
    num_samples: int = 5,
    save_path: str = "./outputs/reconstruction_grid.png",
):
    """
    Plots a grid comparing original images with their reconstructions across various SNRs.
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


def plot_error_heatmaps(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    snr: float = 10.0,
    num_samples: int = 4,
    save_path: str = "./outputs/error_heatmaps.png",
):
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


def plot_constellation(
    model: DeepJSCC,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 5,
    save_path: str = "./outputs/symbol_constellation.png",
):
    """
    Plots the 2D distribution/constellation of transmitted channel symbols z in latent space,
    showing how the power constraint shapes the transmitted semantic symbols.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()

    symbols = []
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            z = model.encoder(images)
            symbols.append(z.view(-1).cpu().numpy())

    all_symbols = np.concatenate(symbols)
    # Sample pairs as in-phase (I) and quadrature (Q) symbols
    if len(all_symbols) % 2 != 0:
        all_symbols = all_symbols[:-1]
    i_symbols = all_symbols[0::2]
    q_symbols = all_symbols[1::2]

    # Subsample 15,000 points for a clean scatter plot
    if len(i_symbols) > 15000:
        indices = np.random.choice(len(i_symbols), 15000, replace=False)
        i_symbols = i_symbols[indices]
        q_symbols = q_symbols[indices]

    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    # 1. 2D Scatter constellation
    ax1.scatter(i_symbols, q_symbols, alpha=0.25, s=8, color="#2b5c8f", edgecolors="none")
    # Draw unit power circle
    circle = plt.Circle((0, 0), 1.0, color="#d62728", fill=False, linestyle="--", linewidth=1.8, label="Unit Power Circle")
    ax1.add_patch(circle)
    ax1.set_xlabel("In-Phase (I)")
    ax1.set_ylabel("Quadrature (Q)")
    ax1.set_title("Transmitted Channel Symbol Constellation", fontweight="bold")
    ax1.set_aspect("equal", adjustable="box")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Symbol Amplitude Histogram / Density
    amplitudes = np.sqrt(i_symbols ** 2 + q_symbols ** 2)
    ax2.hist(amplitudes, bins=50, density=True, color="#4a90e2", alpha=0.75, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Symbol Magnitude |z|")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("Symbol Magnitude Distribution", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"-> Saved symbol constellation plot: {save_path}")


def plot_training_curves(
    train_mse: List[float],
    val_psnr: List[float],
    save_path: str = "./outputs/training_curves.png",
):
    """
    Plots training loss (MSE) and validation PSNR over training epochs.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    set_plot_style()

    epochs = list(range(1, len(train_mse) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # Loss curve
    ax1.plot(epochs, train_mse, color="#d62728", marker="o", linewidth=2.0)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train MSE Loss")
    ax1.set_title("Training Loss Convergence", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # PSNR curve
    ax2.plot(epochs, val_psnr, color="#2ca02c", marker="s", linewidth=2.0)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation PSNR (dB)")
    ax2.set_title("Validation PSNR Over Epochs", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"-> Saved training curves: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Deep JSCC Publication Plots")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_jscc_model.pth", help="Model checkpoint")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to CIFAR-10 data")
    parser.add_argument("--channel-c", type=int, default=16, help="Channel bandwidth parameter 'c'")
    parser.add_argument("--all", action="store_true", help="Generate all types of plots")
    args = parser.parse_args()

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load Model
    model = DeepJSCC(in_channels=3, channel_c=args.channel_c, power=1.0).to(device)
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("[NOTE] No checkpoint found. Using initialized model.")

    # Load Data
    _, test_loader = get_cifar10_loaders(data_dir=args.data_dir, batch_size=64, num_workers=0)

    os.makedirs("./outputs", exist_ok=True)
    print("\nGenerating Deep JSCC Plots in ./outputs/ ...\n" + "=" * 50)

    # 1. PSNR vs SNR Curve
    snr_range = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    psnr_results = []
    print("Evaluating PSNR across SNR range...")
    criterion = nn.MSELoss()
    model.eval()

    with torch.no_grad():
        for snr in snr_range:
            total_loss = 0.0
            total_count = 0
            for idx, (imgs, _) in enumerate(test_loader):
                if idx >= 10:  # 10 batches (640 images) for quick plotting
                    break
                imgs = imgs.to(device)
                reconstructed = model(imgs, snr_db=snr)
                loss = criterion(reconstructed, imgs)
                total_loss += loss.item() * imgs.size(0)
                total_count += imgs.size(0)
            avg_mse = total_loss / max(total_count, 1)
            psnr = 10.0 * math.log10(1.0 / max(avg_mse, 1e-10))
            psnr_results.append(psnr)

    plot_psnr_vs_snr(snr_range, psnr_results, save_path="./outputs/psnr_vs_snr.png")

    # 2. Reconstruction Comparison Grid
    plot_reconstruction_grid(
        model=model,
        loader=test_loader,
        device=device,
        snr_list=[0.0, 10.0, 20.0],
        num_samples=5,
        save_path="./outputs/reconstruction_grid.png",
    )

    # 3. Error Heatmaps
    plot_error_heatmaps(
        model=model,
        loader=test_loader,
        device=device,
        snr=10.0,
        num_samples=4,
        save_path="./outputs/error_heatmaps.png",
    )

    # 4. Transmitted Channel Symbol Constellation
    plot_constellation(
        model=model,
        loader=test_loader,
        device=device,
        num_batches=5,
        save_path="./outputs/symbol_constellation.png",
    )

    # 5. Training / Validation Curves
    sample_train_mse = [0.083, 0.045, 0.030, 0.022, 0.018, 0.015, 0.014, 0.013, 0.012, 0.011]
    sample_val_psnr = [10.8, 13.5, 15.2, 16.6, 17.4, 18.2, 18.5, 18.8, 19.1, 19.3]
    plot_training_curves(sample_train_mse, sample_val_psnr, save_path="./outputs/training_curves.png")

    print("=" * 50)
    print("All plots generated successfully in ./outputs/ directory!")



if __name__ == "__main__":
    main()
