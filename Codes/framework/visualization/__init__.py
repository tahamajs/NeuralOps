"""
Visualization module for attention maps, compression analysis, and reports.
"""

from .base import BaseVisualizer, VISUALIZER_REGISTRY
from .attention_heatmap import AttentionHeatmapVisualizer
from .svd_analysis import SVDAttentionVisualizer
from .report_generator import (
    generate_csv_summary,
    generate_html_report,
    generate_latex_table,
    plot_accuracy_vs_rank,
    plot_radar_chart,
)
from .utils import set_plot_style, save_figure, get_color_palette

__all__ = [
    "BaseVisualizer",
    "VISUALIZER_REGISTRY",
    "AttentionHeatmapVisualizer",
    "SVDAttentionVisualizer",
    "generate_csv_summary",
    "generate_html_report",
    "generate_latex_table",
    "plot_accuracy_vs_rank",
    "plot_radar_chart",
    "set_plot_style",
    "save_figure",
    "get_color_palette",
]