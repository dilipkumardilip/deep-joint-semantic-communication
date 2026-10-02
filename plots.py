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
    parser.add_argument("--dataset", type=str, default=None, choices=["cifar10", "div2k"], help="Dataset: 'cifar10' or 'div2k'")
    parser.add_argument("--arch", type=str, default=None, choices=["baseline", "hd"], help="Model architecture: 'baseline' or 'hd'")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to dataset")
    parser.add_argument("--channel-c", type=int, default=config.CHANNEL_C, help="Channel bandwidth parameter 'c'")
    parser.add_argument("--all", action="store_true", help="Generate all types of plots")
    args = parser.parse_args()

    exp_dir = os.path.join(config.EXPERIMENTS_DIR, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Check for experiment config.json snapshot
    exp_config_file = os.path.join(exp_dir, "config.json")
    exp_cfg = {}
    if os.path.exists(exp_config_file):
        try:
            with open(exp_config_file, "r", encoding="utf-8") as f:
                exp_cfg = json.load(f)
        except Exception as e:
            print(f"Warning: Could not read {exp_config_file}: {e}")

    # Determine dataset
    dataset_name = args.dataset
    if not dataset_name:
        dataset_name = exp_cfg.get("dataset", {}).get("name", "").lower()
        if not dataset_name:
            dataset_name = "cifar10"
        elif "div2k" in dataset_name:
            dataset_name = "div2k"
        else:
            dataset_name = "cifar10"

    # Determine checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path == config.BEST_MODEL_PATH:
        # Check if experiment config specified a custom checkpoint
        cfg_ckpt = exp_cfg.get("paths", {}).get("best_model")
        exp_named_ckpt = os.path.join(config.CHECKPOINT_DIR, f"{args.exp_name}_best_model.pth")
        if cfg_ckpt and os.path.exists(cfg_ckpt):
            checkpoint_path = cfg_ckpt
        elif os.path.exists(exp_named_ckpt):
            checkpoint_path = exp_named_ckpt
        elif dataset_name == "div2k" and os.path.exists(os.path.join(config.CHECKPOINT_DIR, "best_jscc_model_div2k.pth")):
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_jscc_model_div2k.pth")

    # Device
    device = config.get_device()
    # Determine architecture
    arch = args.arch
    if not arch:
        arch = exp_cfg.get("model", {}).get("arch", "baseline")

    print(f"Using device: {device}")
    print(f"Experiment:   {args.exp_name}")
    print(f"Dataset:      {dataset_name}")
    print(f"Architecture: {arch}")
    print(f"Checkpoint:   {checkpoint_path}")

    # Load Model
    model = DeepJSCC(
        in_channels=config.IN_CHANNELS,
        channel_c=args.channel_c,
        power=config.POWER_CONSTRAINT,
        arch=arch,
    ).to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"[NOTE] Checkpoint not found at {checkpoint_path}. Using initialized model.")

    # Load Data
    if dataset_name == "div2k":
        from dataset import get_div2k_loaders
        div2k_dir = args.data_dir or exp_cfg.get("dataset", {}).get("data_dir") or config.DIV2K_HR_DIR
        patch_size = exp_cfg.get("dataset", {}).get("patch_size", config.PATCH_SIZE)
        _, test_loader = get_div2k_loaders(
            hr_dir=div2k_dir,
            patch_size=patch_size,
            batch_size=config.BATCH_SIZE,
            val_batch_size=config.TEST_BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
        )
    else:
        data_dir = args.data_dir or config.DATA_DIR
        _, test_loader = get_cifar10_loaders(data_dir=data_dir, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)

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
            for idx, batch in enumerate(test_loader):
                if idx >= 10:  # 10 batches for quick evaluation
                    break
                imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
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

    # 3. Side-by-side Reconstruction Comparison
    plot_reconstruction_comparison(
        model=model,
        loader=test_loader,
        device=device,
        snr_list=[0.0, 10.0, 20.0],
        num_samples=5,
        save_path=os.path.join(exp_dir, "reconstruction_comparison.png"),
    )

    # 4. Error Heatmaps
    plot_error_heatmaps(
        model=model,
        loader=test_loader,
        device=device,
        snr=config.DEFAULT_SNR_DB,
        num_samples=4,
        save_path=os.path.join(exp_dir, "error_heatmaps.png"),
    )

    # 5. Transmitted Channel Symbol Constellation
    plot_constellation(
        model=model,
        loader=test_loader,
        device=device,
        num_batches=5,
        save_path=os.path.join(exp_dir, "symbol_constellation.png"),
    )

    # 6. Training / Validation Curves (from Experiment history)
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
