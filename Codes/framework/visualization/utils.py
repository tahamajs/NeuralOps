"""
Common plotting utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional


def set_plot_style(style: str = "seaborn-v0_8-darkgrid"):
    """Set global plotting style."""
    plt.style.use(style)
    sns.set_palette("husl")
    sns.set_context("paper", font_scale=1.2)


def save_figure(fig, path: str, dpi: int = 150, bbox_inches: str = "tight"):
    """Save a figure and close it."""
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)


def get_color_palette(n_colors: int, palette: str = "viridis") -> List[str]:
    """Return a list of hex color codes."""
    return sns.color_palette(palette, n_colors).as_hex()


def create_custom_colormap(name: str = "custom_viridis"):
    """Create a custom colormap similar to viridis."""
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#440154", "#3B528B", "#21908C", "#5DC963", "#FDE725"]
    return LinearSegmentedColormap.from_list(name, colors)