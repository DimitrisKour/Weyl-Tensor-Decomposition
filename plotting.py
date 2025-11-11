# plotting.py
"""
Centralized plotting configuration and imports.
Import this at the top of every notebook:
    from plotting import *
"""

# --- core imports ---
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime

# --- rcParams dictionary (your custom style) ---
rcparams = {
    "savefig.bbox": "tight",
    "agg.path.chunksize": 10000,
    "font.family": "serif",
    "font.size": 22,
    "legend.fontsize": 16,
    "legend.loc": "upper right",
    "mathtext.fontset": "stix",

    # X axis
    "xtick.direction": "in",
    "xtick.major.size": 6,
    "xtick.major.width": 1,
    "xtick.minor.size": 3,
    "xtick.minor.width": 1,
    "xtick.minor.visible": True,
    "xtick.top": True,

    # Y axis
    "ytick.direction": "in",
    "ytick.major.size": 6,
    "ytick.major.width": 1,
    "ytick.minor.size": 3,
    "ytick.minor.width": 1,
    "ytick.minor.visible": True,
    "ytick.right": True,

    # Line widths
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,

    # Grid
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.linestyle": "--",
    "grid.color": "k",
    "grid.alpha": 0.5,
    "grid.linewidth": 0.5,

    # Legend
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.fancybox": True,
    "legend.numpoints": 1,

    # Font sizes
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "legend.title_fontsize": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
}

# --- apply to matplotlib globally ---
mpl.rcParams.update(rcparams)

# --- quick-access figure helpers ---
def newfig(size=(7,5)):
    """Create a new figure with default aesthetics."""
    fig, ax = plt.subplots(figsize=size)
    return fig, ax

def savefig(fig, filename, dpi=300):
    """Convenience wrapper for consistent saving."""
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")

class FigureSaver:
    """
    Utility class for saving matplotlib figures in a structured and automatic way.

    Example
    -------
    saver = FigureSaver(base_folder="figures", dpi=300, fmt="png", timestamp=True)
    ...
    plt.plot(x, y)
    saver.save("E2_vs_B2_Schwarzschild")

    This will automatically create: figures/E2_vs_B2_Schwarzschild_2025-11-05_12-30-45.png
    """

    def __init__(self, base_folder="figures", fmt="png", dpi=300, timestamp=False, verbose=True):
        """
        Parameters
        ----------
        base_folder : str
            Root folder where figures will be saved.
        fmt : str
            File format (e.g. 'png', 'pdf', 'svg').
        dpi : int
            Figure resolution.
        timestamp : bool
            Whether to append a timestamp to filenames.
        verbose : bool
            Print save confirmations.
        """
        self.base_folder = base_folder
        self.fmt = fmt
        self.dpi = dpi
        self.timestamp = timestamp
        self.verbose = verbose

        # Create the base folder if it doesn’t exist
        os.makedirs(self.base_folder, exist_ok=True)

    def save(self, filename, subfolder=None, fig=None):
        """
        Save the given matplotlib figure.

        Parameters
        ----------
        filename : str
            Desired filename (without extension).
        subfolder : str, optional
            Subfolder inside base_folder for organizing by category.
        fig : matplotlib.figure.Figure, optional
            The figure to save (default: current active figure).
        """
        # Handle timestamp
        if self.timestamp:
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"{filename}_{timestamp_str}"

        # Build folder path
        save_path = os.path.join(self.base_folder, subfolder or "")
        os.makedirs(save_path, exist_ok=True)

        # Full path
        full_path = os.path.join(save_path, f"{filename}.{self.fmt}")

        # Save figure
        if fig is None:
            fig = plt.gcf()

        fig.savefig(full_path, dpi=self.dpi, bbox_inches="tight")

        if self.verbose:
            print(f"[FigureSaver] Saved figure → {full_path}")

        return full_path