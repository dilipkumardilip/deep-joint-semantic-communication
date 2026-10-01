"""
Dataset Loaders for Deep Joint Source-Channel Communication (Deep JSCC).

Provides clean data loaders for:
  - CIFAR-10 (32×32 low-res benchmark)
  - DIV2K (2K high-resolution benchmark, patch-based training)
"""

import os
import zipfile
from pathlib import Path
from typing import List, Optional, Sized, Tuple, cast

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import config


# ---------------------------------------------------------------------------
# CIFAR-10 Loader (original)
# ---------------------------------------------------------------------------

def get_cifar10_loaders(
    data_dir: str = config.DATA_DIR,
    batch_size: int = config.BATCH_SIZE,
    test_batch_size: int = config.TEST_BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    augment: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns train and test DataLoaders for the CIFAR-10 dataset.

    Images are converted to PyTorch Tensors with values in [0, 1].
    (No mean/std standardization is applied because the Deep JSCC decoder
     uses a Sigmoid output layer to reconstruct pixels in [0, 1]).

    Args:
        data_dir (str): Directory where CIFAR-10 is stored. Default: './data'.
        batch_size (int): Training batch size. Default: 64.
        test_batch_size (int): Testing/validation batch size. Default: 64.
        num_workers (int): DataLoader subprocess workers. Default: 2.
        augment (bool): Whether to apply random horizontal flips for training. Default: False.

    Returns:
        train_loader (DataLoader): DataLoader for 50,000 training images.
        test_loader (DataLoader): DataLoader for 10,000 test images.
    """
    os.makedirs(data_dir, exist_ok=True)

    # Training transforms
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Scale pixel values to [0, 1]
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),  # Scale pixel values to [0, 1]
        ])

    # Testing transforms (deterministic)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# DIV2K High-Resolution Dataset
# ---------------------------------------------------------------------------

class DIV2KDataset(Dataset):
    """
    DIV2K High-Resolution Image Dataset for Deep JSCC training.

    DIV2K contains 800 diverse 2K-resolution RGB training images, making it
    the standard benchmark for image compression and semantic communication papers.

    Strategy:
      - At train time: randomly crop `patch_size × patch_size` patches from each image
        with optional data augmentation (horizontal flip, vertical flip, rotation).
      - At val/test time: use a deterministic center crop for fair comparison.

    Args:
        root_dir (str): Path to the folder containing DIV2K HR images (.png).
        patch_size (int): Spatial size of each extracted square patch (default: 128).
        split (str): One of 'train', 'val', or 'test'.
        train_ratio (float): Fraction of images used for training (default: 0.85).
        augment (bool): Whether to apply random flips/rotations during training.
        seed (int): Random seed used for the train/val split.
    """

    # Supported extensions
    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

    def __init__(
        self,
        root_dir: str,
        patch_size: int = 128,
        split: str = "train",
        train_ratio: float = 0.85,
        augment: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.split = split
        self.augment = augment and (split == "train")

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"DIV2K directory not found: {self.root_dir}\n"
                f"Please download DIV2K_train_HR.zip and extract it to '{self.root_dir}'."
            )

        # Collect all image paths (sorted for reproducibility)
        all_images: List[Path] = sorted([
            p for p in self.root_dir.iterdir()
            if p.suffix.lower() in self.IMG_EXTENSIONS
        ])

        if len(all_images) == 0:
            raise RuntimeError(
                f"No images found in {self.root_dir}. "
                f"Make sure the folder contains .png/.jpg files."
            )

        # Reproducible train/val split
        rng = torch.Generator()
        rng.manual_seed(seed)
        n_total = len(all_images)
        n_train = int(n_total * train_ratio)
        indices = torch.randperm(n_total, generator=rng).tolist()

        if split == "train":
            chosen = [all_images[i] for i in indices[:n_train]]
        elif split in ("val", "test"):
            chosen = [all_images[i] for i in indices[n_train:]]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test'. Got: '{split}'")

        self.image_paths: List[Path] = chosen

        # Base transform: always convert to tensor in [0, 1]
        self._to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_paths)

    def _random_crop(self, img: Image.Image) -> Image.Image:
        """Randomly crop a patch_size × patch_size region from a PIL image."""
        w, h = img.size
        if w < self.patch_size or h < self.patch_size:
            # Upscale small images (shouldn't happen with DIV2K but be safe)
            new_w = max(w, self.patch_size)
            new_h = max(h, self.patch_size)
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            w, h = img.size

        x = int(torch.randint(0, w - self.patch_size + 1, (1,)).item())
        y = int(torch.randint(0, h - self.patch_size + 1, (1,)).item())
        return img.crop((x, y, x + self.patch_size, y + self.patch_size))

    def _center_crop(self, img: Image.Image) -> Image.Image:
        """Deterministic center crop for val/test."""
        w, h = img.size
        if w < self.patch_size or h < self.patch_size:
            new_w = max(w, self.patch_size)
            new_h = max(h, self.patch_size)
            img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
            w, h = img.size
        left = (w - self.patch_size) // 2
        top = (h - self.patch_size) // 2
        return img.crop((left, top, left + self.patch_size, top + self.patch_size))

    def __getitem__(self, index: int) -> torch.Tensor:
        img = Image.open(self.image_paths[index]).convert("RGB")

        # Crop
        if self.split == "train":
            patch = self._random_crop(img)
        else:
            patch = self._center_crop(img)

        patch_tensor: torch.Tensor = self._to_tensor(patch)

        # Augmentation (training only)
        if self.augment:
            if torch.rand(1).item() > 0.5:
                patch_tensor = F.hflip(patch_tensor)
            if torch.rand(1).item() > 0.5:
                patch_tensor = F.vflip(patch_tensor)
            if torch.rand(1).item() > 0.5:
                angle = float(torch.randint(0, 4, (1,)).item() * 90)
                patch_tensor = F.rotate(patch_tensor, angle)

        return patch_tensor


def get_div2k_loaders(
    hr_dir: str = "./data/DIV2K/DIV2K_train_HR",
    patch_size: int = 128,
    train_ratio: float = 0.85,
    batch_size: int = 16,
    val_batch_size: int = 8,
    num_workers: int = config.NUM_WORKERS,
    augment: bool = True,
    seed: int = config.RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns train and val DataLoaders for the DIV2K dataset.

    Args:
        hr_dir (str): Path to the folder with DIV2K HR .png images.
        patch_size (int): Square patch size to extract per image (default: 128).
        train_ratio (float): Fraction of 800 images used for training (default: 0.85 → 680 train / 120 val).
        batch_size (int): Training batch size (default: 16).
        val_batch_size (int): Validation batch size (default: 8).
        num_workers (int): DataLoader worker threads.
        augment (bool): Apply random flips/rotations during training.
        seed (int): Reproducibility seed.

    Returns:
        train_loader, val_loader
    """
    train_dataset = DIV2KDataset(
        root_dir=hr_dir,
        patch_size=patch_size,
        split="train",
        train_ratio=train_ratio,
        augment=augment,
        seed=seed,
    )
    val_dataset = DIV2KDataset(
        root_dir=hr_dir,
        patch_size=patch_size,
        split="val",
        train_ratio=train_ratio,
        augment=False,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader


def unzip_div2k(zip_path: str, extract_to: str) -> str:
    """
    Extracts DIV2K_train_HR.zip to `extract_to` directory.

    Args:
        zip_path (str): Full path to the .zip file.
        extract_to (str): Directory to extract into.

    Returns:
        str: Path to the extracted HR folder.
    """
    print(f"Extracting {zip_path} → {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    extracted = os.path.join(extract_to, "DIV2K_train_HR")
    print(f"Done. HR images at: {extracted}")
    return extracted


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- CIFAR-10 smoke test ---
    print("=" * 60)
    print("CIFAR-10 Loader Test")
    print("=" * 60)
    train_loader, test_loader = get_cifar10_loaders(data_dir="./data", batch_size=16, num_workers=0)
    images, labels = next(iter(train_loader))
    train_count = len(cast(Sized, train_loader.dataset))
    test_count = len(cast(Sized, test_loader.dataset))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch pixel range:  [{images.min().item():.3f}, {images.max().item():.3f}]")
    print(f"Total train batches: {len(train_loader)} ({train_count} images)")
    print(f"Total test batches:  {len(test_loader)} ({test_count} images)")

    # --- DIV2K smoke test (only if data exists) ---
    HR_DIR = "./data/DIV2K/DIV2K_train_HR"
    if os.path.isdir(HR_DIR):
        print("\n" + "=" * 60)
        print("DIV2K Loader Test (128×128 patches)")
        print("=" * 60)
        train_dl, val_dl = get_div2k_loaders(
            hr_dir=HR_DIR, patch_size=128, batch_size=4, val_batch_size=4, num_workers=0
        )
        batch = next(iter(train_dl))
        print(f"Train batch shape: {batch.shape}")
        print(f"Pixel range:       [{batch.min().item():.3f}, {batch.max().item():.3f}]")
        print(f"Train size: {len(cast(Sized, train_dl.dataset))} images")
        print(f"Val size:   {len(cast(Sized, val_dl.dataset))} images")
    else:
        print(f"\n[INFO] DIV2K not found at {HR_DIR}. Download and extract DIV2K_train_HR.zip first.")
