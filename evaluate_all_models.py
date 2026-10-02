"""
Comprehensive Evaluation: All Deep JSCC Models PSNR vs Channel SNR (-5 dB to 25 dB).
Evaluates all trained checkpoints on the exact same test batches and generates
publication-quality unified comparison plots.
"""

import json
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import config
from dataset import get_cifar10_loaders, get_div2k_loaders
from model import DeepJSCC


def prefetch_batches(loader, max_batches=10, device="cpu"):
    batches = []
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        batches.append(imgs.to(device))
    return batches


def evaluate_model_on_batches(model, batches, snrs):
    criterion = nn.MSELoss()
    model.eval()
    psnr_curve = []

    with torch.no_grad():
        for snr in snrs:
            total_loss = 0.0
            total_samples = 0
            for imgs in batches:
                reconstructed = model(imgs, snr_db=snr)
                loss = criterion(reconstructed, imgs)
                total_loss += loss.item() * imgs.size(0)
                total_samples += imgs.size(0)

            avg_mse = total_loss / max(total_samples, 1)
            psnr = 10.0 * math.log10(1.0 / max(avg_mse, 1e-10))
            psnr_curve.append(round(psnr, 2))

    return psnr_curve


def main():
    device = config.get_device()
    print(f"Compute Device: {device}", flush=True)

    snrs = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0]

    models_config = [
        {
            "name": "Exp 2: CIFAR-10 Baseline",
            "short_name": "Exp 2 (CIFAR-10 CNN)",
            "checkpoint": "./checkpoints/experiment_2_best_model.pth",
            "arch": "baseline",
            "dataset": "cifar10",
            "color": "#9b59b6",
            "linestyle": ":",
            "marker": "^",
            "params": 156252,
            "badge": "32×32, 156k params",
        },
        {
            "name": "Exp 3: DIV2K CNN Baseline",
            "short_name": "Exp 3 (DIV2K CNN)",
            "checkpoint": "./checkpoints/experiment_3_best_model.pth",
            "arch": "baseline",
            "dataset": "div2k",
            "color": "#e74c3c",
            "linestyle": "--",
            "marker": "s",
            "params": 156252,
            "badge": "128×128, 156k params",
        },
        {
            "name": "Exp 4: DIV2K ResNet-HD",
            "short_name": "Exp 4 (ResNet-HD)",
            "checkpoint": "./checkpoints/experiment_4_best_model.pth",
            "arch": "hd",
            "dataset": "div2k",
            "color": "#f39c12",
            "linestyle": "-.",
            "marker": "D",
            "params": 2150493,
            "badge": "128×128, 2.15M params",
        },
        {
            "name": "Exp 5: DIV2K Multi-Scale (Ours)",
            "short_name": "Exp 5 (Multi-Scale, SOTA)",
            "checkpoint": "./checkpoints/experiment_5_best_model.pth",
            "arch": "ms",
            "dataset": "div2k",
            "color": "#27ae60",
            "linestyle": "-",
            "marker": "o",
            "params": 608225,
            "badge": "128×128, 608k params (3.5× smaller)",
        },
    ]

    print("\nPreparing and pre-fetching test batches into memory...", flush=True)
    _, val_dl_div2k = get_div2k_loaders(
        hr_dir=config.DIV2K_HR_DIR,
        patch_size=128,
        train_ratio=config.TRAIN_RATIO,
        batch_size=16,
        val_batch_size=16,
        num_workers=0,
        augment=False,
        seed=config.RANDOM_SEED,
        patches_per_image=4,
    )
    div2k_batches = prefetch_batches(val_dl_div2k, max_batches=10, device=device)
    print(f"  DIV2K: {len(div2k_batches)} batches cached on {device}", flush=True)

    _, test_loader_cifar = get_cifar10_loaders(
        data_dir=config.DATA_DIR,
        batch_size=32,
        num_workers=0,
    )
    cifar_batches = prefetch_batches(test_loader_cifar, max_batches=10, device=device)
    print(f"  CIFAR-10: {len(cifar_batches)} batches cached on {device}", flush=True)

    results = {}

    for m_cfg in models_config:
        ckpt_path = m_cfg["checkpoint"]
        if not os.path.exists(ckpt_path):
            print(f"Skipping {m_cfg['name']}: Checkpoint {ckpt_path} not found.", flush=True)
            continue

        print(f"\nEvaluating {m_cfg['name']} ({m_cfg['arch']})...", flush=True)
        model = DeepJSCC(
            in_channels=3,
            channel_c=16,
            power=1.0,
            snr_db=10.0,
            arch=m_cfg["arch"],
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        batches = cifar_batches if m_cfg["dataset"] == "cifar10" else div2k_batches
        psnr_values = evaluate_model_on_batches(model, batches, snrs)
        print(f"  SNRs : {snrs}", flush=True)
        print(f"  PSNRs: {psnr_values}", flush=True)

        results[m_cfg["name"]] = {
            "short_name": m_cfg["short_name"],
            "snrs": snrs,
            "psnrs": psnr_values,
            "color": m_cfg["color"],
            "linestyle": m_cfg["linestyle"],
            "marker": m_cfg["marker"],
            "params": m_cfg["params"],
            "badge": m_cfg["badge"],
            "dataset": m_cfg["dataset"],
            "arch": m_cfg["arch"],
        }

    # Save JSON results
    out_json = "experiments/all_models_psnr_vs_snr.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation metrics to: {out_json}", flush=True)

    # ---------------------------------------------------------------------------
    # Plot 1: Combined Side-by-Side Publication Figure
    # ---------------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(15, 6.2), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.22)

    # Panel A: DIV2K HD Head-to-Head Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    div2k_keys = [k for k, v in results.items() if v["dataset"] == "div2k"]

    for k in div2k_keys:
        d = results[k]
        lw = 3.0 if "Multi-Scale" in k else 2.0
        ms = 8 if "Multi-Scale" in k else 6
        ax1.plot(
            d["snrs"],
            d["psnrs"],
            label=f"{d['short_name']} [{d['badge']}]",
            color=d["color"],
            linestyle=d["linestyle"],
            marker=d["marker"],
            linewidth=lw,
            markersize=ms,
            alpha=0.95,
        )

    # Shaded improvement area between Exp 5 and Exp 4
    if "Exp 5: DIV2K Multi-Scale (Ours)" in results and "Exp 4: DIV2K ResNet-HD" in results:
        snr_vals = results["Exp 5: DIV2K Multi-Scale (Ours)"]["snrs"]
        p5 = results["Exp 5: DIV2K Multi-Scale (Ours)"]["psnrs"]
        p4 = results["Exp 4: DIV2K ResNet-HD"]["psnrs"]
        ax1.fill_between(
            snr_vals, p4, p5,
            color="#27ae60",
            alpha=0.15,
            label="Gain over ResNet-HD (+1.4 dB)",
        )

    ax1.set_title("A. High-Definition (DIV2K 128×128) Deep JSCC Comparison", fontsize=12.5, fontweight="bold", pad=12)
    ax1.set_xlabel("Wireless Channel SNR (dB)", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Reconstruction PSNR (dB)", fontsize=11, fontweight="semibold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9.5)
    ax1.set_xlim(-6, 26)

    # Panel B: All Experiments Unified (CIFAR-10 + DIV2K Models)
    ax2 = fig.add_subplot(gs[0, 1])

    for k, d in results.items():
        lw = 2.8 if "Multi-Scale" in k else 1.8
        ms = 7 if "Multi-Scale" in k else 5
        ax2.plot(
            d["snrs"],
            d["psnrs"],
            label=f"{d['short_name']}",
            color=d["color"],
            linestyle=d["linestyle"],
            marker=d["marker"],
            linewidth=lw,
            markersize=ms,
        )

    ax2.set_title("B. All Deep JSCC Architectures Rate-Distortion", fontsize=12.5, fontweight="bold", pad=12)
    ax2.set_xlabel("Wireless Channel SNR (dB)", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Reconstruction PSNR (dB)", fontsize=11, fontweight="semibold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9.5)
    ax2.set_xlim(-6, 26)

    fig.suptitle("Deep Joint Source-Channel Communication: PSNR vs Channel SNR Across All Experiments", fontsize=14, fontweight="bold", y=0.98)
    
    save_path1 = "experiments/all_experiments_psnr_vs_snr.png"
    plt.savefig(save_path1, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved side-by-side comparative plot to: {save_path1}", flush=True)

    # ---------------------------------------------------------------------------
    # Plot 2: Single Unified Clean Curve Plot (Everything in One Single Plot)
    # ---------------------------------------------------------------------------
    plt.figure(figsize=(10, 6.2), dpi=300)
    for k, d in results.items():
        lw = 3.2 if "Multi-Scale" in k else 2.0
        ms = 8 if "Multi-Scale" in k else 6
        plt.plot(
            d["snrs"],
            d["psnrs"],
            label=f"{k} [{d['badge']}]",
            color=d["color"],
            linestyle=d["linestyle"],
            marker=d["marker"],
            linewidth=lw,
            markersize=ms,
        )

    plt.title("Deep JSCC: All Experiments PSNR vs Channel SNR (-5 dB to 25 dB)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Channel SNR (dB)", fontsize=11, fontweight="semibold")
    plt.ylabel("Reconstruction PSNR (dB)", fontsize=11, fontweight="semibold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9.5)
    plt.xlim(-6, 26)
    plt.tight_layout()

    save_path2 = "experiments/psnr_vs_snr_all_unified.png"
    plt.savefig(save_path2, dpi=300)
    plt.close()
    print(f"Saved single unified plot to: {save_path2}", flush=True)

    # Also copy to static folder for web app or direct viewing
    os.makedirs("outputs", exist_ok=True)
    os.system(f"cp {save_path1} outputs/all_experiments_psnr_vs_snr.png")
    os.system(f"cp {save_path2} outputs/psnr_vs_snr_all_unified.png")
    print("Copies saved to outputs/ as well.", flush=True)


if __name__ == "__main__":
    main()
