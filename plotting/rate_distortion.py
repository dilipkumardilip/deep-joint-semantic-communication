"""
Rate-Distortion and Graceful Degradation Visualization for Deep JSCC.
"""

import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

import config
from .style import set_plot_style


def plot_psnr_vs_snr(
    snr_list: List[float],
    psnr_list: List[float],
    title: str = "Deep JSCC: PSNR vs Channel SNR",
    save_path: str = os.path.join(config.OUTPUTS_DIR, "psnr_vs_snr.png"),
    benchmark_data: Optional[Dict[str, List[float]]] = None,
) -> None:
    """
    Plots the Deep JSCC Rate-Distortion curve across different Channel SNRs.
    Demonstrates graceful degradation without the cliff effect of classical separation schemes.

    Args:
        snr_list (List[float]): List of evaluated channel SNRs (in dB).
        psnr_list (List[float]): Corresponding PSNR values (in dB).
        title (str): Title for the figure.
        save_path (str): Destination file path for saving PNG.
        benchmark_data (Dict, optional): Dictionary of comparison baselines (e.g., BPG+LDPC).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    set_plot_style()

    plt.figure(figsize=(8, 5.5))
    plt.plot(
        snr_list,
        psnr_list,
        marker="o",
        color="#1f77b4",
        linewidth=2.5,
        markersize=7,
        label="Deep JSCC (Proposed)",
    )

    if benchmark_data:
        colors = ["#ff7f0e", "#2ca02c", "#d62728"]
        for idx, (bench_name, bench_psnr) in enumerate(benchmark_data.items()):
            color = colors[idx % len(colors)]
            plt.plot(
                snr_list[:len(bench_psnr)],
                bench_psnr,
                marker="s",
                linestyle="--",
                linewidth=1.8,
                color=color,
                label=bench_name,
            )

    plt.xlabel("Channel SNR (dB)")
    plt.ylabel("Reconstruction PSNR (dB)")
    plt.title(title, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"-> Saved PSNR vs SNR plot: {save_path}")
