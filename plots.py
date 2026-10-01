"""
Unified Plotting CLI Runner for Deep Joint Source-Channel Communication (Deep JSCC).

Dispatches plotting routines from the modular `plots/` package:
- Rate Distortion: `plot_psnr_vs_snr`
- Reconstructions: `plot_reconstruction_grid`
- Error Heatmaps:  `plot_error_heatmaps`
- Constellations:  `plot_constellation`
- Convergence:     `plot_training_curves`

Usage:
    python plots.py --all --exp-name experiment_2
"""

import argparse
import json
import math
import os
import torch
import torch.nn as nn

import config
from dataset import get_cifar10_loaders
from model import DeepJSCC
from plotting import (
    plot_constellation,
    plot_error_heatmaps,
    plot_psnr_vs_snr,
    plot_reconstruction_comparison,
    plot_reconstruction_grid,
    plot_training_curves,
    set_plot_style,
)

__all__ = [
    "plot_constellation",
    "plot_error_heatmaps",
    "plot_psnr_vs_snr",
    "plot_reconstruction_comparison",
    "plot_reconstruction_grid",
    "plot_training_curves",
    "set_plot_style",
]


def main():
    parser = argparse.ArgumentParser(description="Generate Deep JSCC Publication Plots")
    parser.add_argument("--exp-name", type=str, default=config.DEFAULT_EXP_NAME, help="Experiment name (e.g. experiment_2)")
    parser.add_argument("--checkpoint", type=str, default=config.BEST_MODEL_PATH, help="Model checkpoint")
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR, help="Path to CIFAR-10 data")
    parser.add_argument("--channel-c", type=int, default=config.CHANNEL_C, help="Channel bandwidth parameter 'c'")
    parser.add_argument("--all", action="store_true", help="Generate all types of plots")
    args = parser.parse_args()

    exp_dir = os.path.join(config.EXPERIMENTS_DIR, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Device
    device = config.get_device()
    print(f"Using device: {device}")

    # Load Model
    model = DeepJSCC(
        in_channels=config.IN_CHANNELS,
        channel_c=args.channel_c,
        power=config.POWER_CONSTRAINT,
    ).to(device)

    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("[NOTE] No checkpoint found. Using initialized model.")

    # Load Data
    _, test_loader = get_cifar10_loaders(data_dir=args.data_dir, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

    print(f"\nGenerating Deep JSCC Plots in {exp_dir}/ ...\n" + "=" * 50)

    # 1. PSNR vs SNR Curve
    snr_range = config.TEST_SNRS
    psnr_results = []
    print("Evaluating PSNR across SNR range...")
    criterion = nn.MSELoss()
    model.eval()

    with torch.no_grad():
        for snr in snr_range:
            total_loss = 0.0
            total_count = 0
            for idx, (imgs, _) in enumerate(test_loader):
                if idx >= 10:  # 10 batches for quick evaluation
                    break
                imgs = imgs.to(device)
                reconstructed = model(imgs, snr_db=snr)
                loss = criterion(reconstructed, imgs)
                total_loss += loss.item() * imgs.size(0)
                total_count += imgs.size(0)
            avg_mse = total_loss / max(total_count, 1)
            psnr = 10.0 * math.log10(1.0 / max(avg_mse, 1e-10))
            psnr_results.append(psnr)

    plot_psnr_vs_snr(snr_range, psnr_results, save_path=os.path.join(exp_dir, "psnr_vs_snr.png"))

    # 2. Reconstruction Comparison Grid
    plot_reconstruction_grid(
        model=model,
        loader=test_loader,
        device=device,
        snr_list=[0.0, 10.0, 20.0],
        num_samples=5,
        save_path=os.path.join(exp_dir, "reconstruction_grid.png"),
    )

    # 3. Error Heatmaps
    plot_error_heatmaps(
        model=model,
        loader=test_loader,
        device=device,
        snr=config.DEFAULT_SNR_DB,
        num_samples=4,
        save_path=os.path.join(exp_dir, "error_heatmaps.png"),
    )

    # 4. Transmitted Channel Symbol Constellation
    plot_constellation(
        model=model,
        loader=test_loader,
        device=device,
        num_batches=5,
        save_path=os.path.join(exp_dir, "symbol_constellation.png"),
    )

    # 5. Training / Validation Curves (from Experiment history)
    history_file = os.path.join(exp_dir, "history.json")
    hist_dict = None
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            hist_dict = json.load(f)

    plot_training_curves(history_dict=hist_dict, save_path=os.path.join(exp_dir, "training_curves.png"))

    print("=" * 50)
    print(f"All plots generated successfully in {exp_dir}/ directory!")


if __name__ == "__main__":
    main()
