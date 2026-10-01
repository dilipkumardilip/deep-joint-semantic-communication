"""
Plot Styling and Theme Configurations for Deep JSCC Visualizations.
"""

import matplotlib.pyplot as plt


def set_plot_style() -> None:
    """Configures clean, publication-ready Matplotlib aesthetics."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
    })
