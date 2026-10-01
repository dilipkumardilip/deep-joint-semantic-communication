"""
Training Pipeline for Deep Joint Source-Channel Communication (Deep JSCC) on Images.

Features:
- Splits CIFAR-10 training data into 70% Training and 30% Validation sets.
- Trains end-to-end (Encoder -> AWGN Channel -> Decoder) using MSE Loss.
- Evaluates reconstruction quality using Peak Signal-to-Noise Ratio (PSNR in dB).
- Automatically selects Apple Silicon MPS / CUDA / CPU.
- Saves the best model checkpoint based on validation PSNR.
"""

import argparse
import datetime
import json
import math
import os
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

import config
from model import DeepJSCC
from plotting import plot_training_curves
from utils.metrics import calculate_psnr
from utils.reporting import format_duration, generate_experiment_report, save_history_json


def get_cifar10_70_percent_split(
    data_dir: str = config.DATA_DIR,
    train_ratio: float = config.TRAIN_SPLIT,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads CIFAR-10 and splits the 50,000 training images into:
      - 70% Training set (35,000 images)
      - 30% Validation set (15,000 images)
    Also loads the 10,000 testing images.

    Args:
        data_dir (str): Path to data directory.
        train_ratio (float): Ratio for training split (default: 0.7 for 70%).
        batch_size (int): Batch size for loaders.
        num_workers (int): DataLoader worker threads.
        seed (int): Random seed for reproducible splitting.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Transform: scale pixel values to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    total_train_len = len(full_train_dataset)
    train_len = int(total_train_len * train_ratio)
    val_len = total_train_len - train_len

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_len, val_len],
        generator=generator,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    snr_db: float,
) -> Tuple[float, float]:
    """
    Runs one training epoch over the training set.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, _ in train_loader:
        images = images.to(device)
        batch_size = images.size(0)

        # Forward pass: Encode -> Wireless Channel -> Decode
        reconstructed = model(images, snr_db=snr_db)

        # Compute MSE loss between original and reconstructed image
        loss = criterion(reconstructed, images)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_mse = running_loss / total_samples
    epoch_psnr = calculate_psnr(epoch_mse)
    return epoch_mse, epoch_psnr


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    snr_db: float,
) -> Tuple[float, float]:
    """
    Evaluates the model on validation or test set without gradient computation.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            batch_size = images.size(0)

            reconstructed = model(images, snr_db=snr_db)
            loss = criterion(reconstructed, images)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_mse = running_loss / total_samples
    avg_psnr = calculate_psnr(avg_mse)
    return avg_mse, avg_psnr


def main():
    parser = argparse.ArgumentParser(description="Train Deep JSCC Image Semantic Communication Model")
    parser.add_argument("--exp-name", type=str, default="experiment_2", help="Experiment identifier name")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate for Adam optimizer")
    parser.add_argument("--eta-min", type=float, default=1e-5, help="Minimum learning rate for Cosine Annealing")
    parser.add_argument("--channel-c", type=int, default=config.CHANNEL_C, help="Number of latent channel features 'c'")
    parser.add_argument("--snr", type=float, default=config.DEFAULT_SNR_DB, help="Channel Signal-to-Noise Ratio (SNR) in dB")
    parser.add_argument("--train-split", type=float, default=config.TRAIN_SPLIT, help="Fraction of data used for training")
    parser.add_argument("--save-dir", type=str, default=config.CHECKPOINT_DIR, help="Directory to save model checkpoints")
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR, help="Directory where CIFAR-10 data is stored")
    args = parser.parse_args()

    # 1. Device configuration
    device = config.get_device()
    print(f"Using compute device: {device}")

    # Create Experiment directory
    exp_dir = os.path.join("./experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # 2. Data Preparation (70% Train, 30% Validation)
    print(f"\nLoading CIFAR-10 dataset ({int(args.train_split * 100)}% Train / {int((1 - args.train_split) * 100)}% Val)...")
    train_loader, val_loader, test_loader = get_cifar10_70_percent_split(
        data_dir=args.data_dir,
        train_ratio=args.train_split,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
    )
    print(f"Training set:   {len(train_loader.dataset)} images ({len(train_loader)} batches)")  # type: ignore[arg-type]
    print(f"Validation set: {len(val_loader.dataset)} images ({len(val_loader)} batches)")      # type: ignore[arg-type]
    print(f"Test set:       {len(test_loader.dataset)} images ({len(test_loader)} batches)")    # type: ignore[arg-type]

    # 3. Model, Loss, Optimizer, LR Scheduler
    print(f"\nInitializing Deep JSCC Model for {args.exp_name} (c={args.channel_c}, SNR={args.snr} dB, Epochs={args.epochs})...")
    model = DeepJSCC(
        in_channels=config.IN_CHANNELS,
        channel_c=args.channel_c,
        power=config.POWER_CONSTRAINT,
        snr_db=args.snr,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    
    # Cosine Annealing LR Scheduler decaying from initial LR to eta_min across all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

    # 4. Training Loop
    start_total_time = time.time()
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
    print(f"{'Epoch':<7} {'Current LR':<12} {'Train MSE':<12} {'Train PSNR':<13} {'Val MSE':<12} {'Val PSNR':<12} {'Time':<8} {'Status'}")
    print("=" * 90)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_mse, train_psnr = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            snr_db=args.snr,
        )

        val_mse, val_psnr = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            snr_db=args.snr,
        )

        # Step the LR scheduler
        scheduler.step()
        epoch_duration = round(time.time() - epoch_start, 2)

        # Record history
        history["epoch"].append(epoch)
        history["lr"].append(current_lr)
        history["train_mse"].append(train_mse)
        history["train_psnr"].append(train_psnr)
        history["val_mse"].append(val_mse)
        history["val_psnr"].append(val_psnr)
        history["epoch_time"].append(epoch_duration)

        # Checkpoint saving
        is_best = val_psnr > best_val_psnr
        status = ""
        if is_best:
            best_val_psnr = val_psnr
            ckpt_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_psnr": val_psnr,
                "val_mse": val_mse,
                "channel_c": args.channel_c,
                "snr_db": args.snr,
                "lr": current_lr,
            }
            torch.save(ckpt_dict, best_model_path)
            torch.save(ckpt_dict, default_best_path)
            status = "* Best"

        print(
            f"{epoch:<7} {current_lr:<12.6f} {train_mse:<12.5f} {train_psnr:<13.2f} {val_mse:<12.5f} {val_psnr:<12.2f} {epoch_duration:>5.1f}s  {status}"
        )

    total_training_time = round(time.time() - start_total_time, 2)
    history["total_time_seconds"] = total_training_time
    history["avg_epoch_time_seconds"] = round(total_training_time / max(args.epochs, 1), 2)

    total_min = int(total_training_time // 60)
    total_sec = int(total_training_time % 60)
    print("=" * 90)
    print(f"Training completed in {total_min}m {total_sec}s ({total_training_time:.1f}s total, avg {history['avg_epoch_time_seconds']:.1f}s/epoch)!")
    print(f"Best Validation PSNR: {best_val_psnr:.2f} dB")
    print(f"Saved best model checkpoint to: {best_model_path}")

    # Save history json
    history_file = os.path.join(exp_dir, "history.json")
    save_history_json(history, history_file)
    print(f"Saved training history to: {history_file}")

    # Plot training convergence curves
    curve_path = os.path.join(exp_dir, "training_curves.png")
    plot_training_curves(history_dict=history, save_path=curve_path)
    print(f"Saved training curve plot to: {curve_path}")

    # 5. Final Evaluation on 10,000 Test Images
    print("\nRunning final evaluation on held-out Test set...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_mse, test_psnr = evaluate(model, test_loader, criterion, device, snr_db=args.snr)
    print(f"Final Test MSE:  {test_mse:.5f}")
    print(f"Final Test PSNR: {test_psnr:.2f} dB")

    # 6. Generate detailed Experiment Markdown file inside the experiment folder
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
    print(f"\n[SUCCESS] Generated experiment report: {md_file}")

    # 7. Save exact run configuration snapshot inside the experiment folder
    exp_config = {
        "experiment_name": args.exp_name,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "model": {
            "in_channels": config.IN_CHANNELS,
            "channel_c": args.channel_c,
            "power": config.POWER_CONSTRAINT,
            "image_size": list(config.IMG_SIZE),
            "transmitted_symbols_k": args.channel_c * 8 * 8,
        },
        "dataset": {
            "data_dir": args.data_dir,
            "dataset_name": "CIFAR-10",
            "train_split": args.train_split,
            "val_split": round(1.0 - args.train_split, 2),
            "batch_size": args.batch_size,
            "num_workers": config.NUM_WORKERS,
            "random_seed": config.RANDOM_SEED,
        },
        "channel": {
            "channel_type": "AWGN",
            "train_snr_db": args.snr,
            "test_snrs": config.TEST_SNRS,
        },
        "training": {
            "epochs": args.epochs,
            "initial_lr": args.lr,
            "eta_min": args.eta_min,
            "weight_decay": config.WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "loss_criterion": "MSELoss",
            "optimizer": "Adam",
        },
        "results_summary": {
            "best_val_psnr_db": round(best_val_psnr, 2),
            "test_mse": round(test_mse, 5),
            "test_psnr_db": round(test_psnr, 2),
            "total_training_duration": format_duration(total_training_time),
            "total_training_seconds": total_training_time,
            "avg_epoch_seconds": history["avg_epoch_time_seconds"],
        },
        "paths": {
            "checkpoint_dir": args.save_dir,
            "best_model_path": best_model_path,
            "experiment_dir": exp_dir,
        },
    }
    exp_config_path = os.path.join(exp_dir, "config.json")
    with open(exp_config_path, "w", encoding="utf-8") as f:
        json.dump(exp_config, f, indent=2)
    print(f"[SUCCESS] Saved experiment configuration snapshot: {exp_config_path}")



if __name__ == "__main__":
    main()
