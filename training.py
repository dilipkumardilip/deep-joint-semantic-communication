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
import math
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

from model import DeepJSCC


def calculate_psnr(mse: float, max_val: float = 1.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Formula: PSNR = 10 * log10(max_val^2 / MSE)
    """
    if mse <= 1e-10:
        return 100.0  # Perfect reconstruction
    return 10.0 * math.log10((max_val ** 2) / mse)


def get_cifar10_70_percent_split(
    data_dir: str = "./data",
    train_ratio: float = 0.7,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
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
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--channel-c", type=int, default=16, help="Number of latent channel features 'c'")
    parser.add_argument("--snr", type=float, default=10.0, help="Channel Signal-to-Noise Ratio (SNR) in dB")
    parser.add_argument("--train-split", type=float, default=0.7, help="Fraction of data used for training (default: 0.7 = 70%)")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory where CIFAR-10 data is stored")
    args = parser.parse_args()

    # 1. Device configuration (MPS for Apple Silicon, CUDA for Nvidia, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using compute device: {device}")

    # 2. Data Preparation (70% Train, 30% Validation)
    print(f"\nLoading CIFAR-10 dataset ({int(args.train_split * 100)}% Train / {int((1 - args.train_split) * 100)}% Val)...")
    train_loader, val_loader, test_loader = get_cifar10_70_percent_split(
        data_dir=args.data_dir,
        train_ratio=args.train_split,
        batch_size=args.batch_size,
        num_workers=2,
    )
    print(f"Training set:   {len(train_loader.dataset)} images ({len(train_loader)} batches)")  # type: ignore[arg-type]
    print(f"Validation set: {len(val_loader.dataset)} images ({len(val_loader)} batches)")      # type: ignore[arg-type]
    print(f"Test set:       {len(test_loader.dataset)} images ({len(test_loader)} batches)")    # type: ignore[arg-type]

    # 3. Model, Loss, Optimizer
    print(f"\nInitializing Deep JSCC Model (c={args.channel_c}, SNR={args.snr} dB)...")
    model = DeepJSCC(
        in_channels=3,
        channel_c=args.channel_c,
        power=1.0,
        snr_db=args.snr,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. Training Loop
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_psnr = -1.0
    best_model_path = os.path.join(args.save_dir, "best_jscc_model.pth")

    print("\n" + "=" * 70)
    print(f"{'Epoch':<8} {'Train MSE':<12} {'Train PSNR':<14} {'Val MSE':<12} {'Val PSNR':<12} {'Status'}")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
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

        scheduler.step()

        # Checkpoint saving
        is_best = val_psnr > best_val_psnr
        status = ""
        if is_best:
            best_val_psnr = val_psnr
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_psnr": val_psnr,
                    "val_mse": val_mse,
                    "channel_c": args.channel_c,
                    "snr_db": args.snr,
                },
                best_model_path,
            )
            status = "* Saved Best"

        print(
            f"{epoch:<8} {train_mse:<12.5f} {train_psnr:<14.2f} {val_mse:<12.5f} {val_psnr:<12.2f} {status}"
        )

    print("=" * 70)
    print(f"Training completed! Best Validation PSNR: {best_val_psnr:.2f} dB")
    print(f"Saved best model checkpoint to: {best_model_path}")

    # 5. Final Evaluation on 10,000 Test Images
    print("\nRunning final evaluation on held-out Test set...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_mse, test_psnr = evaluate(model, test_loader, criterion, device, snr_db=args.snr)
    print(f"Final Test MSE:  {test_mse:.5f}")
    print(f"Final Test PSNR: {test_psnr:.2f} dB")


if __name__ == "__main__":
    main()
