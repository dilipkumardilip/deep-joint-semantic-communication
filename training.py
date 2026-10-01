"""
Training Pipeline for Deep Joint Source-Channel Communication (Deep JSCC) on Images.

Features:
- Supports both CIFAR-10 (32×32) and DIV2K (128×128 HD patches) datasets.
- Trains end-to-end (Encoder -> AWGN Channel -> Decoder) using MSE Loss.
- Evaluates reconstruction quality using Peak Signal-to-Noise Ratio (PSNR in dB).
- Automatically selects Apple Silicon MPS / CUDA / CPU.
- Saves the best model checkpoint based on validation PSNR.
- Saves per-experiment config snapshot and Markdown report inside the experiment folder.

Usage:
    python training.py --dataset div2k --exp-name experiment_3
    python training.py --dataset cifar10 --exp-name experiment_4
"""

import argparse
import datetime
import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

import config
from dataset import get_div2k_loaders, DIV2KDataset
from model import DeepJSCC
from plotting import plot_training_curves
from utils.metrics import calculate_psnr
from utils.reporting import format_duration, generate_experiment_report, save_history_json


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_cifar10_70_percent_split(
    data_dir: str = config.DATA_DIR,
    train_ratio: float = config.TRAIN_SPLIT,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads CIFAR-10 and splits the 50,000 training images into:
      - 70% Training set  (35,000 images)
      - 30% Validation set (15,000 images)
    Also loads the 10,000 testing images separately.
    """
    transform = transforms.Compose([transforms.ToTensor()])

    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    total_train_len = len(full_train_dataset)
    train_len = int(total_train_len * train_ratio)
    val_len = total_train_len - train_len

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_train_dataset, [train_len, val_len], generator=generator
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    return train_loader, val_loader, test_loader


def get_div2k_split(
    hr_dir: str = config.DIV2K_HR_DIR,
    patch_size: int = config.PATCH_SIZE,
    train_ratio: float = config.TRAIN_RATIO,
    batch_size: int = config.BATCH_SIZE,
    val_batch_size: int = config.TEST_BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    augment: bool = config.AUGMENT,
    seed: int = config.RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads DIV2K HR images and splits into train / val loaders (patch-based).
    Returns (train_loader, val_loader, val_loader) — DIV2K has no separate test set,
    so val loader is returned as the test loader as well.
    """
    train_dl, val_dl = get_div2k_loaders(
        hr_dir=hr_dir,
        patch_size=patch_size,
        train_ratio=train_ratio,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        augment=augment,
        seed=seed,
    )
    return train_dl, val_dl, val_dl  # val used as test too


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    snr_db: float,
    dataset_name: str = "cifar10",
) -> Tuple[float, float]:
    """Runs one training epoch over the training set."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in train_loader:
        # DIV2K returns raw tensors; CIFAR-10 returns (images, labels) tuples
        images = batch if isinstance(batch, torch.Tensor) else batch[0]
        images = images.to(device)
        batch_sz = images.size(0)

        reconstructed = model(images, snr_db=snr_db)
        loss = criterion(reconstructed, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_sz
        total_samples += batch_sz

    epoch_mse = running_loss / total_samples
    epoch_psnr = calculate_psnr(epoch_mse)
    return epoch_mse, epoch_psnr


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    snr_db: float,
    dataset_name: str = "cifar10",
) -> Tuple[float, float]:
    """Evaluates the model on a DataLoader without gradient computation."""
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch if isinstance(batch, torch.Tensor) else batch[0]
            images = images.to(device)
            batch_sz = images.size(0)

            reconstructed = model(images, snr_db=snr_db)
            loss = criterion(reconstructed, images)

            running_loss += loss.item() * batch_sz
            total_samples += batch_sz

    avg_mse = running_loss / total_samples
    avg_psnr = calculate_psnr(avg_mse)
    return avg_mse, avg_psnr


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Deep JSCC Image Semantic Communication Model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=config.DATASET_NAME,
        choices=["cifar10", "div2k"],
        help="Dataset to train on: 'cifar10' (32×32) or 'div2k' (HD 128×128 patches). Default: from config.json",
    )
    parser.add_argument("--exp-name", type=str, default="experiment_3", help="Experiment identifier")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--eta-min", type=float, default=1e-6, help="Min LR for Cosine Annealing")
    parser.add_argument("--channel-c", type=int, default=config.CHANNEL_C, help="Latent channel feature count 'c'")
    parser.add_argument("--snr", type=float, default=config.DEFAULT_SNR_DB, help="Training SNR in dB")
    parser.add_argument("--patch-size", type=int, default=config.PATCH_SIZE, help="Patch size for DIV2K (ignored for CIFAR-10)")
    parser.add_argument("--train-split", type=float, default=config.TRAIN_SPLIT, help="Train fraction (CIFAR-10 only)")
    parser.add_argument("--train-ratio", type=float, default=config.TRAIN_RATIO, help="Train fraction (DIV2K only: 0.85 → 680/120)")
    parser.add_argument("--save-dir", type=str, default=config.CHECKPOINT_DIR, help="Checkpoint directory")
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR, help="Base data directory")
    parser.add_argument("--div2k-dir", type=str, default=config.DIV2K_HR_DIR, help="DIV2K HR images folder")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation for DIV2K training")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Device
    # -----------------------------------------------------------------------
    device = config.get_device()
    print(f"Using compute device: {device}")

    # -----------------------------------------------------------------------
    # 2. Experiment directories
    # -----------------------------------------------------------------------
    exp_dir = os.path.join("./experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 3. Data loaders
    # -----------------------------------------------------------------------
    dataset_label: str
    patch_size_used: int

    if args.dataset == "div2k":
        print(f"\nLoading DIV2K HR dataset from: {args.div2k_dir}")
        print(f"  Patch size : {args.patch_size}×{args.patch_size}")
        print(f"  Train ratio: {args.train_ratio:.0%} train / {1-args.train_ratio:.0%} val")
        train_loader, val_loader, test_loader = get_div2k_split(
            hr_dir=args.div2k_dir,
            patch_size=args.patch_size,
            train_ratio=args.train_ratio,
            batch_size=args.batch_size,
            val_batch_size=config.TEST_BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            augment=not args.no_augment,
            seed=config.RANDOM_SEED,
        )
        dataset_label = "DIV2K"
        patch_size_used = args.patch_size
    else:
        print(f"\nLoading CIFAR-10 dataset ({int(args.train_split * 100)}% Train / {int((1-args.train_split)*100)}% Val)...")
        train_loader, val_loader, test_loader = get_cifar10_70_percent_split(
            data_dir=args.data_dir,
            train_ratio=args.train_split,
            batch_size=args.batch_size,
            num_workers=config.NUM_WORKERS,
        )
        dataset_label = "CIFAR-10"
        patch_size_used = 32

    print(f"  Training set  : {len(train_loader.dataset)} images ({len(train_loader)} batches)")   # type: ignore[arg-type]
    print(f"  Validation set: {len(val_loader.dataset)} images ({len(val_loader)} batches)")       # type: ignore[arg-type]

    # -----------------------------------------------------------------------
    # 4. Model, loss, optimizer, scheduler
    # -----------------------------------------------------------------------
    print(f"\nInitializing Deep JSCC (c={args.channel_c}, SNR={args.snr} dB, Epochs={args.epochs})...")
    model = DeepJSCC(
        in_channels=config.IN_CHANNELS,
        channel_c=args.channel_c,
        power=config.POWER_CONSTRAINT,
        snr_db=args.snr,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

    # -----------------------------------------------------------------------
    # 5. Training loop
    # -----------------------------------------------------------------------
    start_total = time.time()
    best_val_psnr = -1.0
    best_model_path = os.path.join(args.save_dir, f"{args.exp_name}_best_model.pth")
    default_best_path = config.BEST_MODEL_PATH

    history: Dict[str, Any] = {
        "epoch": [],
        "lr": [],
        "train_mse": [],
        "train_psnr": [],
        "val_mse": [],
        "val_psnr": [],
        "epoch_time": [],
    }

    print("\n" + "=" * 90)
    print(f"{'Epoch':<7} {'LR':<12} {'Train MSE':<12} {'Train PSNR':<13} {'Val MSE':<12} {'Val PSNR':<12} {'Time':<8} {'Status'}")
    print("=" * 90)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_mse, train_psnr = train_one_epoch(
            model=model, train_loader=train_loader, optimizer=optimizer,
            criterion=criterion, device=device, snr_db=args.snr,
            dataset_name=args.dataset,
        )
        val_mse, val_psnr = evaluate(
            model=model, loader=val_loader, criterion=criterion,
            device=device, snr_db=args.snr, dataset_name=args.dataset,
        )

        scheduler.step()
        epoch_dur = round(time.time() - epoch_start, 2)

        history["epoch"].append(epoch)
        history["lr"].append(current_lr)
        history["train_mse"].append(train_mse)
        history["train_psnr"].append(train_psnr)
        history["val_mse"].append(val_mse)
        history["val_psnr"].append(val_psnr)
        history["epoch_time"].append(epoch_dur)

        status = ""
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_psnr": val_psnr,
                "val_mse": val_mse,
                "channel_c": args.channel_c,
                "snr_db": args.snr,
                "lr": current_lr,
                "dataset": args.dataset,
                "patch_size": patch_size_used,
            }
            torch.save(ckpt, best_model_path)
            torch.save(ckpt, default_best_path)
            status = "★ Best"

        print(
            f"{epoch:<7} {current_lr:<12.6f} {train_mse:<12.5f} {train_psnr:<13.2f}"
            f" {val_mse:<12.5f} {val_psnr:<12.2f} {epoch_dur:>5.1f}s  {status}"
        )

    total_time = round(time.time() - start_total, 2)
    history["total_time_seconds"] = total_time
    history["avg_epoch_time_seconds"] = round(total_time / max(args.epochs, 1), 2)

    print("=" * 90)
    print(f"Training done in {format_duration(total_time)}  (avg {history['avg_epoch_time_seconds']:.1f}s/epoch)")
    print(f"Best Validation PSNR: {best_val_psnr:.2f} dB → checkpoint: {best_model_path}")

    # -----------------------------------------------------------------------
    # 6. Final test evaluation
    # -----------------------------------------------------------------------
    print("\nRunning final evaluation on held-out set...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_mse, test_psnr = evaluate(model, test_loader, criterion, device, snr_db=args.snr, dataset_name=args.dataset)
    print(f"Final Test MSE:  {test_mse:.5f}")
    print(f"Final Test PSNR: {test_psnr:.2f} dB")

    # -----------------------------------------------------------------------
    # 7. Save history JSON
    # -----------------------------------------------------------------------
    history_file = os.path.join(exp_dir, "history.json")
    save_history_json(history, history_file)
    print(f"Saved training history → {history_file}")

    # -----------------------------------------------------------------------
    # 8. Plot training curves
    # -----------------------------------------------------------------------
    curve_path = os.path.join(exp_dir, "training_curves.png")
    plot_training_curves(history_dict=history, save_path=curve_path)
    print(f"Saved training curves  → {curve_path}")

    # -----------------------------------------------------------------------
    # 9. Markdown experiment report (inside experiment folder)
    # -----------------------------------------------------------------------
    md_file = os.path.join(exp_dir, f"{args.exp_name}.md")
    generate_experiment_report(
        exp_name=args.exp_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        initial_lr=args.lr,
        eta_min=args.eta_min,
        channel_c=args.channel_c,
        snr_db=args.snr,
        train_split=args.train_split,
        best_val_psnr=best_val_psnr,
        test_mse=test_mse,
        test_psnr=test_psnr,
        history=history,
        save_path=md_file,
    )
    print(f"[SUCCESS] Experiment report → {md_file}")

    # -----------------------------------------------------------------------
    # 10. config.json snapshot inside experiment folder
    # -----------------------------------------------------------------------
    exp_config: Dict[str, Any] = {
        "experiment_name": args.exp_name,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "model": {
            "in_channels": config.IN_CHANNELS,
            "channel_c": args.channel_c,
            "power": config.POWER_CONSTRAINT,
            "image_size": [patch_size_used, patch_size_used],
            "transmitted_symbols_k": args.channel_c * (patch_size_used // 4) * (patch_size_used // 4),
        },
        "dataset": {
            "name": dataset_label,
            "data_dir": args.data_dir if args.dataset == "cifar10" else args.div2k_dir,
            "patch_size": patch_size_used,
            "train_split": args.train_split if args.dataset == "cifar10" else args.train_ratio,
            "batch_size": args.batch_size,
            "num_workers": config.NUM_WORKERS,
            "augment": not args.no_augment if args.dataset == "div2k" else False,
            "random_seed": config.RANDOM_SEED,
        },
        "channel": {
            "type": "AWGN",
            "train_snr_db": args.snr,
            "test_snrs": config.TEST_SNRS,
        },
        "training": {
            "epochs": args.epochs,
            "initial_lr": args.lr,
            "eta_min": args.eta_min,
            "weight_decay": config.WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "loss": "MSELoss",
            "optimizer": "Adam",
        },
        "results": {
            "best_val_psnr_db": round(best_val_psnr, 2),
            "test_mse": round(test_mse, 5),
            "test_psnr_db": round(test_psnr, 2),
            "total_duration": format_duration(total_time),
            "total_seconds": total_time,
            "avg_epoch_seconds": history["avg_epoch_time_seconds"],
        },
        "paths": {
            "checkpoint_dir": args.save_dir,
            "best_model": best_model_path,
            "experiment_dir": exp_dir,
        },
    }
    exp_config_path = os.path.join(exp_dir, "config.json")
    with open(exp_config_path, "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)
    print(f"[SUCCESS] Config snapshot → {exp_config_path}")


if __name__ == "__main__":
    main()
