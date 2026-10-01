"""
Transmitted Channel Symbol Constellation & Latent Distribution for Deep JSCC.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from model import DeepJSCC
from plotting.style import set_plot_style


def plot_constellation(
    model: DeepJSCC,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 5,
    save_path: str = os.path.join(config.OUTPUTS_DIR, "symbol_constellation.png"),
) -> None:
    """
    Plots the 2D distribution/constellation of transmitted channel symbols z in latent space,
    showing how the power constraint shapes the transmitted semantic symbols.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()

    symbols = []
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            images = images.to(device)
            z = model.encoder(images)
            symbols.append(z.view(-1).cpu().numpy())

    all_symbols = np.concatenate(symbols)
    # Sample pairs as in-phase (I) and quadrature (Q) symbols
    if len(all_symbols) % 2 != 0:
        all_symbols = all_symbols[:-1]
    i_symbols = all_symbols[0::2]
    q_symbols = all_symbols[1::2]

    # Subsample 15,000 points for a clean scatter plot
    if len(i_symbols) > 15000:
        indices = np.random.choice(len(i_symbols), 15000, replace=False)
        i_symbols = i_symbols[indices]
        q_symbols = q_symbols[indices]

    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    # 1. 2D Scatter constellation
    ax1.scatter(i_symbols, q_symbols, alpha=0.25, s=8, color="#2b5c8f", edgecolors="none")
    # Draw unit power circle
    circle = plt.Circle((0, 0), 1.0, color="#d62728", fill=False, linestyle="--", linewidth=1.8, label="Unit Power Circle")
    ax1.add_patch(circle)
    ax1.set_xlabel("In-Phase (I)")
    ax1.set_ylabel("Quadrature (Q)")
    ax1.set_title("Transmitted Channel Symbol Constellation", fontweight="bold")
    ax1.set_aspect("equal", adjustable="box")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Symbol Amplitude Histogram / Density
    amplitudes = np.sqrt(i_symbols ** 2 + q_symbols ** 2)
    ax2.hist(amplitudes, bins=50, density=True, color="#4a90e2", alpha=0.75, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Symbol Magnitude |z|")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("Symbol Magnitude Distribution", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"-> Saved symbol constellation plot: {save_path}")
