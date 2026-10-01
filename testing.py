"""
Testing & Evaluation Script for Deep Joint Source-Channel Communication (Deep JSCC).

Features:
- Evaluates the trained Deep JSCC model on the remaining 30% data (15,000 images).
- Computes Test MSE and Peak Signal-to-Noise Ratio (PSNR in dB).
- Evaluates performance across multiple Channel SNRs (e.g., 0 dB to 20 dB).
- Saves visual comparisons (Original vs Reconstructed images) to `./experiments/<exp_name>/reconstruction_comparison.png`.
"""

import argparse
import math
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

import config
from model import DeepJSCC
from plotting import plot_reconstruction_comparison
from utils.metrics import calculate_psnr


def get_30_percent_test_loader(
    data_dir: str = config.DATA_DIR,
    split_ratio: float = config.TRAIN_SPLIT,
    batch_size: int = config.TEST_BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.RANDOM_SEED,
) -> DataLoader:
    """
    Loads CIFAR-10 and extracts the 30% held-out split (15,000 images)
    matching the exact split used during training.

    Args:
        data_dir (str): Path to data directory.
        split_ratio (float): Training split ratio (0.7), meaning 0.3 (30%) is for evaluation.
        batch_size (int): Batch size.
        num_workers (int): DataLoader worker threads.
        seed (int): Seed matching the training split.

    Returns:
        test_loader (DataLoader): DataLoader for the 30% evaluation split.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    total_len = len(full_train_dataset)
    train_len = int(total_len * split_ratio)
    eval_30_len = total_len - train_len

    generator = torch.Generator().manual_seed(seed)
    _, eval_subset = random_split(
        full_train_dataset,
        [train_len, eval_30_len],
        generator=generator,
    )

    eval_loader = DataLoader(
        eval_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return eval_loader


def evaluate_snr(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    snr_db: float,
) -> Tuple[float, float]:
    """
    Evaluates the model over the evaluation dataset at a specific channel SNR.
    """
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            batch_size = images.size(0)

            reconstructed = model(images, snr_db=snr_db)
            loss = criterion(reconstructed, images)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_mse = total_loss / total_samples
    avg_psnr = calculate_psnr(avg_mse)
    return avg_mse, avg_psnr


def main():
    parser = argparse.ArgumentParser(description="Test Deep JSCC Model on 30% Evaluation Split")
    parser.add_argument("--exp-name", type=str, default=config.DEFAULT_EXP_NAME, help="Experiment name (e.g., experiment_2)")
    parser.add_argument("--checkpoint", type=str, default=config.BEST_MODEL_PATH, help="Path to trained model checkpoint")
    parser.add_argument("--channel-c", type=int, default=config.CHANNEL_C, help="Channel bandwidth parameter 'c'")
    parser.add_argument("--batch-size", type=int, default=config.TEST_BATCH_SIZE, help="Batch size for evaluation")
    parser.add_argument("--data-dir", type=str, default=config.DATA_DIR, help="Directory where CIFAR-10 data is stored")
    parser.add_argument("--snr-sweep", action="store_true", help="Evaluate across multiple SNRs")
    parser.add_argument("--save-plot", type=str, default=None, help="Path to save comparison image")
    args = parser.parse_args()

    exp_dir = os.path.join(config.EXPERIMENTS_DIR, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    save_plot_path = args.save_plot or os.path.join(exp_dir, "reconstruction_comparison.png")

    # 1. Device selection
    device = config.get_device()
    print(f"Using compute device: {device}")

    # 2. Load 30% Data Split
    print("\nLoading 30% held-out evaluation dataset (15,000 images)...")
    eval_loader = get_30_percent_test_loader(
        data_dir=args.data_dir,
        split_ratio=config.TRAIN_SPLIT,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
    )
    print(f"Evaluation set loaded: {len(eval_loader.dataset)} images ({len(eval_loader)} batches)")  # type: ignore[arg-type]

    # 3. Model setup & checkpoint loading
    model = DeepJSCC(
        in_channels=config.IN_CHANNELS,
        channel_c=args.channel_c,
        power=config.POWER_CONSTRAINT,
    ).to(device)

    trained_snr = config.DEFAULT_SNR_DB
    if os.path.exists(args.checkpoint):
        print(f"\nLoading weights from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        trained_snr = checkpoint.get("snr_db", config.DEFAULT_SNR_DB)
        print(f"Model trained at SNR = {trained_snr} dB, best val PSNR = {checkpoint.get('val_psnr', 'N/A'):.2f} dB")
    else:
        print(f"\n[WARNING] Checkpoint not found at {args.checkpoint}. Running evaluation with initialized weights.")

    # 4. Evaluation at nominal SNR
    print("\n" + "=" * 55)
    print(f"Evaluation on 30% Split @ Nominal SNR = {trained_snr} dB")
    print("=" * 55)
    mse, psnr = evaluate_snr(model, eval_loader, device=device, snr_db=trained_snr)
    print(f"Mean Squared Error (MSE):  {mse:.5f}")
    print(f"Reconstruction PSNR:       {psnr:.2f} dB")
    print("=" * 55)

    # 5. Multi-SNR Robustness Evaluation (Deep JSCC Characteristic Curve)
    test_snrs = [s for s in config.TEST_SNRS if s in [0.0, 5.0, 10.0, 15.0, 20.0]] or [0.0, 5.0, 10.0, 15.0, 20.0]
    print("\n" + "-" * 55)
    print("Performance across Wireless Channel SNRs (PSNR vs SNR):")
    print("-" * 55)
    print(f"{'Channel SNR (dB)':<20} {'MSE':<15} {'PSNR (dB)':<15}")
    print("-" * 55)

    for snr in test_snrs:
        curr_mse, curr_psnr = evaluate_snr(model, eval_loader, device=device, snr_db=snr)
        print(f"{snr:<20.1f} {curr_mse:<15.5f} {curr_psnr:<15.2f}")
    print("-" * 55)

    # 6. Save visual comparison grid using modular plotting package
    plot_reconstruction_comparison(
        model=model,
        loader=eval_loader,
        device=device,
        snr_list=[0.0, 10.0, 20.0],
        save_path=save_plot_path,
        num_samples=5,
    )
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
