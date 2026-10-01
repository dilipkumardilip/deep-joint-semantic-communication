"""
Testing & Evaluation Script for Deep Joint Source-Channel Communication (Deep JSCC).

Features:
- Evaluates the trained Deep JSCC model on the remaining 30% data (15,000 images).
- Computes Test MSE and Peak Signal-to-Noise Ratio (PSNR in dB).
- Evaluates performance across multiple Channel SNRs (e.g., 0 dB to 20 dB).
- Saves visual comparisons (Original vs Reconstructed images) to `./outputs/reconstruction_comparison.png`.
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

from model import DeepJSCC


def calculate_psnr(mse: float, max_val: float = 1.0) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    Formula: PSNR = 10 * log10(max_val^2 / MSE)
    """
    if mse <= 1e-10:
        return 100.0
    return 10.0 * math.log10((max_val ** 2) / mse)


def get_30_percent_test_loader(
    data_dir: str = "./data",
    split_ratio: float = 0.7,
    batch_size: int = 64,
    num_workers: int = 2,
    seed: int = 42,
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


def save_visual_comparison(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    snr_list: List[float],
    save_path: str = "./outputs/reconstruction_comparison.png",
    num_samples: int = 5,
):
    """
    Saves a side-by-side visual comparison of original vs reconstructed images
    transmitted over channels with different SNR values.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()

    # Get a batch of images
    images, _ = next(iter(loader))
    images = images[:num_samples].to(device)

    # Reconstruct images for each SNR in snr_list
    reconstructions = {}
    with torch.no_grad():
        for snr in snr_list:
            reconstructions[snr] = model(images, snr_db=snr).cpu().clamp(0.0, 1.0)

    images = images.cpu()

    # Plot
    cols = 1 + len(snr_list)
    rows = num_samples
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))

    if rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(rows):
        # Column 0: Original
        orig_img = images[i].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(orig_img)
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("Original", fontsize=12, fontweight="bold")

        # Subsequent columns: Reconstructions at different SNRs
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


def main():
    parser = argparse.ArgumentParser(description="Test Deep JSCC Model on 30% Evaluation Split")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_jscc_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--channel-c", type=int, default=16, help="Channel bandwidth parameter 'c'")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory where CIFAR-10 data is stored")
    parser.add_argument("--snr-sweep", action="store_true", help="Evaluate across multiple SNRs (0, 5, 10, 15, 20 dB)")
    parser.add_argument("--save-plot", type=str, default="./outputs/reconstruction_comparison.png", help="Path to save comparison image")
    args = parser.parse_args()

    # 1. Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using compute device: {device}")

    # 2. Load 30% Data Split
    print("\nLoading 30% held-out evaluation dataset (15,000 images)...")
    eval_loader = get_30_percent_test_loader(
        data_dir=args.data_dir,
        split_ratio=0.7,
        batch_size=args.batch_size,
        num_workers=2,
    )
    print(f"Evaluation set loaded: {len(eval_loader.dataset)} images ({len(eval_loader)} batches)")  # type: ignore[arg-type]

    # 3. Model setup & checkpoint loading
    model = DeepJSCC(in_channels=3, channel_c=args.channel_c, power=1.0).to(device)

    trained_snr = 10.0
    if os.path.exists(args.checkpoint):
        print(f"\nLoading weights from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        trained_snr = checkpoint.get("snr_db", 10.0)
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
    test_snrs = [0.0, 5.0, 10.0, 15.0, 20.0]
    print("\n" + "-" * 55)
    print("Performance across Wireless Channel SNRs (PSNR vs SNR):")
    print("-" * 55)
    print(f"{'Channel SNR (dB)':<20} {'MSE':<15} {'PSNR (dB)':<15}")
    print("-" * 55)

    for snr in test_snrs:
        curr_mse, curr_psnr = evaluate_snr(model, eval_loader, device=device, snr_db=snr)
        print(f"{snr:<20.1f} {curr_mse:<15.5f} {curr_psnr:<15.2f}")
    print("-" * 55)

    # 6. Save visual comparison grid
    save_visual_comparison(
        model=model,
        loader=eval_loader,
        device=device,
        snr_list=[0.0, 10.0, 20.0],
        save_path=args.save_plot,
        num_samples=5,
    )
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
