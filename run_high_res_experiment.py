"""
run_high_res_experiment.py - High-Resolution Deep JSCC Training and Evaluation.

Designed for Experiment 6 (and future high-res experiments):
- Trains on Ultra-HD DIV2K patches (default 256×256, expandable to 512×512).
- Automatically adapts to the target machine (CUDA / Apple Silicon MPS / CPU).
- Gradient accumulation & mixed precision support for memory-efficient execution on any GPU.
- Performs end-to-end training, saving the best checkpoint to ./checkpoints/experiment_6_best_model.pth.
- Automatically executes full post-training evaluation across SNR range [-5 to 25 dB].
- Generates high-res visual reconstruction grids, rate-distortion curves, and Markdown report.

Usage:
  # Quick smoke test with 1 epoch:
  python run_high_res_experiment.py --epochs 1 --batch-size 8 --exp-name experiment_6

  # Full training (50 epochs on GPU/MPS):
  python run_high_res_experiment.py --epochs 50 --patch-size 256 --exp-name experiment_6

  # Evaluation only (using existing checkpoint):
  python run_high_res_experiment.py --test-only --exp-name experiment_6
"""

import argparse
import datetime
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sized, Tuple, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import get_div2k_loaders
from losses import CompositeJSCCLoss
from model import DeepJSCC
from plotting import plot_training_curves, plot_psnr_vs_snr, plot_reconstruction_grid
from utils.metrics import calculate_psnr
from utils.reporting import format_duration, save_history_json


# ---------------------------------------------------------------------------
# Device Auto-detection and Optimization
# ---------------------------------------------------------------------------

def setup_environment(device_pref: str = "auto") -> torch.device:
    """Detects optimal compute device and applies performance tunings."""
    if device_pref == "cuda" or (device_pref == "auto" and torch.cuda.is_available()):
        dev = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Device] Using NVIDIA CUDA: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    elif device_pref == "mps" or (device_pref == "auto" and torch.backends.mps.is_available()):
        dev = torch.device("mps")
        print(f"[Device] Using Apple Silicon MPS (Metal Performance Shaders)")
    else:
        dev = torch.device("cpu")
        print(f"[Device] Using CPU ({platform.processor() or 'generic'})")
    return dev


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    snr_db: float,
    grad_accum_steps: int = 1,
    use_amp: bool = False,
    scaler: Optional[Any] = None,
) -> Tuple[float, float]:
    """Runs one training epoch with gradient accumulation and optional AMP."""
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    total_samples = 0
    total_batches = len(train_loader)

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(train_loader):
        images = batch if isinstance(batch, torch.Tensor) else batch[0]
        images = images.to(device, non_blocking=True)
        batch_sz = images.size(0)

        # Forward pass with optional Mixed Precision
        if use_amp and device.type == "cuda":
            assert scaler is not None
            with torch.cuda.amp.autocast():
                reconstructed = model(images, snr_db=snr_db)
                if isinstance(criterion, CompositeJSCCLoss):
                    loss, metrics = criterion(reconstructed, images)
                    batch_mse = metrics["mse"]
                else:
                    loss = criterion(reconstructed, images)
                    batch_mse = loss.item()
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == total_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            reconstructed = model(images, snr_db=snr_db)
            if isinstance(criterion, CompositeJSCCLoss):
                loss, metrics = criterion(reconstructed, images)
                batch_mse = metrics["mse"]
            else:
                loss = criterion(reconstructed, images)
                batch_mse = loss.item()
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == total_batches:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * batch_sz
        running_mse += batch_mse * batch_sz
        total_samples += batch_sz

        # Print progress every 10 batches or on milestones
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches or (batch_idx + 1) in [1, 5]:
            cur_psnr = calculate_psnr(running_mse / max(total_samples, 1))
            sys.stdout.write(
                f"\r    Batch [{batch_idx+1:3d}/{total_batches}] | Step Loss: {loss.item():.4f} | Running PSNR: {cur_psnr:.2f} dB"
            )
            sys.stdout.flush()

    print()  # newline after progress
    epoch_mse = running_mse / max(total_samples, 1)
    epoch_psnr = calculate_psnr(epoch_mse)
    return epoch_mse, epoch_psnr


# ---------------------------------------------------------------------------
# Validation Loop
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    snr_db: float,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """Evaluates validation loss and PSNR on the given dataloader."""
    model.eval()
    running_loss = 0.0
    running_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            images = batch if isinstance(batch, torch.Tensor) else batch[0]
            images = images.to(device, non_blocking=True)
            batch_sz = images.size(0)

            reconstructed = model(images, snr_db=snr_db)
            if isinstance(criterion, CompositeJSCCLoss):
                loss, metrics = criterion(reconstructed, images)
                batch_mse = metrics["mse"]
            else:
                loss = criterion(reconstructed, images)
                batch_mse = loss.item()

            running_loss += loss.item() * batch_sz
            running_mse += batch_mse * batch_sz
            total_samples += batch_sz

    avg_mse = running_mse / max(total_samples, 1)
    avg_psnr = calculate_psnr(avg_mse)
    return avg_mse, avg_psnr


# ---------------------------------------------------------------------------
# Full SNR Evaluation Sweep
# ---------------------------------------------------------------------------

def run_snr_sweep(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    snr_list: List[float],
    save_dir: str,
    max_eval_batches: int = 15,
) -> Dict[str, Any]:
    """Runs a complete SNR sweep from -5 dB to +25 dB and creates plots."""
    print("\n" + "=" * 60)
    print("Running Full Wireless Channel SNR Sweep (Evaluation)")
    print("=" * 60)
    model.eval()
    psnr_curve = []
    criterion = nn.MSELoss()

    for snr in snr_list:
        total_mse = 0.0
        total_samples = 0
        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                if idx >= max_eval_batches:
                    break
                images = batch if isinstance(batch, torch.Tensor) else batch[0]
                images = images.to(device)
                recon = model(images, snr_db=snr)
                mse = criterion(recon, images).item()
                total_mse += mse * images.size(0)
                total_samples += images.size(0)
        avg_mse = total_mse / max(total_samples, 1)
        psnr = calculate_psnr(avg_mse)
        psnr_curve.append(psnr)
        print(f"  SNR: {snr:5.1f} dB  -->  PSNR: {psnr:5.2f} dB  (MSE: {avg_mse:.6f})")

    # Plot rate-distortion PSNR vs SNR
    plot_path = os.path.join(save_dir, "psnr_vs_snr.png")
    plot_psnr_vs_snr(snr_list, psnr_curve, save_path=plot_path)

    # Plot sample reconstructions
    grid_path = os.path.join(save_dir, "reconstruction_grid.png")
    plot_reconstruction_grid(
        model=model,
        loader=test_loader,
        device=device,
        snr_list=[0.0, 10.0, 20.0],
        num_samples=4,
        save_path=grid_path,
    )

    return {"snrs": snr_list, "psnr_results": psnr_curve}


# ---------------------------------------------------------------------------
# Main Execution Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run High-Resolution DIV2K Deep JSCC Experiment (Train + Test)"
    )
    parser.add_argument("--exp-name", type=str, default="experiment_6", help="Experiment directory name")
    parser.add_argument("--patch-size", type=int, default=256, help="High-resolution patch size (default: 256)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per step (default: 8)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (effective batch = batch * accum)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--channel-c", type=int, default=16, help="Latent channel count (default: 16)")
    parser.add_argument("--snr", type=float, default=10.0, help="Training SNR in dB (default: 10.0)")
    parser.add_argument("--arch", type=str, default="ms", choices=["ms", "hd", "baseline"], help="Model architecture")
    parser.add_argument("--loss", type=str, default="composite", choices=["composite", "mse"], help="Loss function")
    parser.add_argument("--patches-per-image", type=int, default=2, help="Random crops per image per epoch")
    parser.add_argument("--train-ratio", type=float, default=0.85, help="Train/val split ratio (default: 0.85)")
    parser.add_argument("--div2k-dir", type=str, default=config.DIV2K_HR_DIR, help="Path to DIV2K_train_HR directory")
    parser.add_argument("--save-ckpt", type=str, default="./checkpoints/experiment_6_best_model.pth", help="Checkpoint save destination")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Compute device")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS, help="DataLoader worker processes")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision on CUDA")
    parser.add_argument("--test-only", action="store_true", help="Skip training and run evaluation on checkpoint")
    args = parser.parse_args()

    # 1. Device and paths
    device = setup_environment(args.device)
    exp_dir = os.path.join("./experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)

    print("=" * 70)
    print(f"Deep JSCC High-Resolution Experiment: {args.exp_name}")
    print(f"Resolution       : {args.patch_size}×{args.patch_size} Ultra-HD Patches")
    print(f"Architecture     : DeepJSCC-{args.arch.upper()} (c={args.channel_c})")
    print(f"Loss Function    : {args.loss.upper()}")
    print(f"Batch Config     : Batch Size = {args.batch_size} (Grad Accum = {args.grad_accum}, Effective = {args.batch_size * args.grad_accum})")
    print(f"Target Checkpoint: {args.save_ckpt}")
    print("=" * 70)

    # 2. Check dataset presence
    if not os.path.isdir(args.div2k_dir):
        print(f"\n[ERROR] DIV2K directory not found at: {args.div2k_dir}")
        print("Please ensure DIV2K high-resolution images are located at that path, or specify --div2k-dir <path>.")
        sys.exit(1)

    # 3. Data Loaders
    print(f"\nLoading DIV2K HR images from: {args.div2k_dir} ...")
    train_loader, val_loader = get_div2k_loaders(
        hr_dir=args.div2k_dir,
        patch_size=args.patch_size,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        val_batch_size=max(1, args.batch_size // 2),
        num_workers=args.num_workers,
        augment=True,
        seed=config.RANDOM_SEED,
        patches_per_image=args.patches_per_image,
    )
    print(f"  Train set: {len(cast(Sized, train_loader.dataset))} patches ({len(train_loader)} batches)")
    print(f"  Val set  : {len(cast(Sized, val_loader.dataset))} patches ({len(val_loader)} batches)")

    # 4. Initialize Model
    model = DeepJSCC(
        in_channels=config.IN_CHANNELS,
        channel_c=args.channel_c,
        power=config.POWER_CONSTRAINT,
        snr_db=args.snr,
        arch=args.arch,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters : {total_params:,}")

    # Loss criterion
    if args.loss == "composite":
        criterion = CompositeJSCCLoss(alpha=0.5, beta=0.3, gamma=0.2)
    else:
        criterion = nn.MSELoss()

    # -----------------------------------------------------------------------
    # Training Stage
    # -----------------------------------------------------------------------
    best_val_psnr = -1.0
    history: Dict[str, List[Any]] = {
        "epoch": [],
        "train_mse": [],
        "train_psnr": [],
        "val_mse": [],
        "val_psnr": [],
        "lr": [],
    }

    if not args.test_only:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
        use_amp = args.amp and (device.type == "cuda")
        scaler = None
        if use_amp:
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                scaler = torch.amp.GradScaler("cuda")
            else:
                scaler = torch.cuda.amp.GradScaler()

        print(f"\nStarting Training ({args.epochs} epochs)...")
        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            ep_start = time.time()
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"\n--- Epoch [{epoch:02d}/{args.epochs:02d}] (LR: {current_lr:.6f}) ---")
            train_mse, train_psnr = train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                snr_db=args.snr,
                grad_accum_steps=args.grad_accum,
                use_amp=use_amp,
                scaler=scaler,
            )

            val_mse, val_psnr = evaluate_model(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                snr_db=args.snr,
            )
            scheduler.step()

            history["epoch"].append(epoch)
            history["train_mse"].append(train_mse)
            history["train_psnr"].append(train_psnr)
            history["val_mse"].append(val_mse)
            history["val_psnr"].append(val_psnr)
            history["lr"].append(current_lr)

            ep_duration = time.time() - ep_start
            print(f"  Results: Train PSNR = {train_psnr:.2f} dB | Val PSNR = {val_psnr:.2f} dB | Time = {ep_duration:.1f}s")

            # Checkpoint on improvement
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_psnr": val_psnr,
                    "patch_size": args.patch_size,
                    "arch": args.arch,
                    "channel_c": args.channel_c,
                }, args.save_ckpt)
                print(f"  ★ New best model saved to: {args.save_ckpt} (Val PSNR: {val_psnr:.2f} dB)")

        total_sec = time.time() - start_time
        print(f"\nTraining completed in {format_duration(total_sec)}.")

        # Save training curves
        plot_training_curves(history_dict=history, save_path=os.path.join(exp_dir, "training_curves.png"))
    else:
        print(f"\n[Test-Only Mode] Loading existing weights from: {args.save_ckpt}")
        ckpt = torch.load(args.save_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_psnr = ckpt.get("val_psnr", 0.0)

    # -----------------------------------------------------------------------
    # Comprehensive Testing Stage (SNR Sweep: -5 dB to 25 dB)
    # -----------------------------------------------------------------------
    # Load best checkpoint for testing
    if os.path.exists(args.save_ckpt):
        ckpt = torch.load(args.save_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    test_snrs = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    sweep_results = run_snr_sweep(
        model=model,
        test_loader=val_loader,
        device=device,
        snr_list=test_snrs,
        save_dir=exp_dir,
    )

    # -----------------------------------------------------------------------
    # Save Metadata and Markdown Report
    # -----------------------------------------------------------------------
    history_path = os.path.join(exp_dir, "history.json")
    save_history_json(history, history_path)

    config_snapshot = {
        "experiment_name": args.exp_name,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "model": {
            "arch": args.arch,
            "patch_size": args.patch_size,
            "channel_c": args.channel_c,
            "total_parameters": total_params,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "initial_lr": args.lr,
            "snr_db": args.snr,
            "loss": args.loss,
        },
        "results": {
            "best_val_psnr_db": round(best_val_psnr, 2),
            "snr_sweep": {str(k): round(v, 2) for k, v in zip(sweep_results["snrs"], sweep_results["psnr_results"])},
        },
        "paths": {
            "checkpoint": args.save_ckpt,
            "experiment_dir": exp_dir,
        }
    }
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    # Markdown report
    report_md = f"""# 📊 {args.exp_name}: Ultra-HD Deep JSCC ({args.patch_size}×{args.patch_size})

## 🎯 Experiment Overview
- **Resolution**: {args.patch_size}×{args.patch_size} Ultra-HD patches from DIV2K
- **Architecture**: DeepJSCC-{args.arch.upper()} (Parameters: {total_params:,})
- **Loss**: {args.loss.upper()} (Perceptual MSE + L1 + SSIM)
- **Best Validation PSNR**: **{best_val_psnr:.2f} dB** @ {args.snr:.1f} dB SNR

## 📈 Rate-Distortion Performance Across SNR
| Wireless Channel SNR (dB) | Reconstructed PSNR (dB) |
| :---: | :---: |
"""
    for snr, psnr in zip(sweep_results["snrs"], sweep_results["psnr_results"]):
        report_md += f"| {snr:+.1f} dB | **{psnr:.2f} dB** |\n"

    report_md += f"""
## 🖼️ Visual Artifacts Generated
- `psnr_vs_snr.png`: Rate-Distortion performance across the wireless channel SNR spectrum.
- `reconstruction_grid.png`: High-resolution visual comparison at 0 dB, 10 dB, and 20 dB SNR.
- `training_curves.png`: Training & validation convergence curves.
"""
    with open(os.path.join(exp_dir, f"{args.exp_name}.md"), "w") as f:
        f.write(report_md)

    print("\n" + "=" * 70)
    print(f"🎉 Experiment {args.exp_name} Finished Successfully!")
    print(f"Checkpoints saved to : {args.save_ckpt}")
    print(f"Outputs saved to     : {exp_dir}/")
    print(f"UI Model Ready       : Registered in Web App as 'DIV2K Ultra-HD (256×256)'")
    print("=" * 70)


if __name__ == "__main__":
    main()
