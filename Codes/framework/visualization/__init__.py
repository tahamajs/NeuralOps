# framework/visualization/__init__.py
from .base import BaseVisualizer, VISUALIZER_REGISTRY
from .attention_heatmap import AttentionHeatmapVisualizer
from .svd_analysis import SVDAttentionVisualizer
from .utils import set_plot_style, save_figure, get_color_palette, create_custom_colormap
from .report_generator import (
    generate_csv_summary,
    generate_html_report,
    generate_latex_table,
    plot_accuracy_vs_rank,
    plot_radar_chart,
)

__all__ = [
    "BaseVisualizer",
    "VISUALIZER_REGISTRY",
    "AttentionHeatmapVisualizer",
    "SVDAttentionVisualizer",
    "set_plot_style",
    "save_figure",
    "get_color_palette",
    "create_custom_colormap",
    "generate_csv_summary",
    "generate_html_report",
    "generate_latex_table",
    "plot_accuracy_vs_rank",
    "plot_radar_chart",
]