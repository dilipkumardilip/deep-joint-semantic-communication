"""
CIFAR-10 Dataset Loader for Deep Joint Source-Channel Communication (Deep JSCC).

Provides clean data loaders for training and evaluating image semantic communication models.
"""

import os
from typing import Tuple
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    test_batch_size: int = 64,
    num_workers: int = 2,
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


from typing import Sized, Tuple, cast

if __name__ == "__main__":
    train_loader, test_loader = get_cifar10_loaders(data_dir="./data", batch_size=16, num_workers=0)
    images, labels = next(iter(train_loader))
    train_count = len(cast(Sized, train_loader.dataset))
    test_count = len(cast(Sized, test_loader.dataset))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch pixel range:  [{images.min().item():.3f}, {images.max().item():.3f}]")
    print(f"Total train batches: {len(train_loader)} ({train_count} images)")
    print(f"Total test batches:  {len(test_loader)} ({test_count} images)")


