# framework/visualization/utils.py
"""
Helper functions for visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional


def set_plot_style(style: str = 'seaborn-v0_8-darkgrid'):
    """
    Set consistent plotting style for all visualizations.
    
    Args:
        style: Matplotlib style name
    """
    plt.style.use(style)
    sns.set_palette("husl")
    sns.set_context("paper", font_scale=1.2)


def save_figure(fig, path: str, dpi: int = 150, bbox_inches: str = 'tight'):
    """
    Save figure with consistent settings.
    
    Args:
        fig: Matplotlib figure
        path: Output file path
        dpi: Resolution
        bbox_inches: Bounding box setting
    """
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)


def get_color_palette(n_colors: int, palette: str = 'viridis') -> List[str]:
    """
    Get color palette with n colors.
    
    Args:
        n_colors: Number of colors
        palette: Seaborn palette name
        
    Returns:
        List of color hex codes
    """
    return sns.color_palette(palette, n_colors).as_hex()