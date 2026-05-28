from .base import BaseVisualizer, VISUALIZER_REGISTRY, Registry
from .plot_registry import PLOT_REGISTRY, register_default_plots
from .utils import set_plot_style, save_figure, get_color_palette, create_custom_colormap
from .report_generator import (
    generate_csv_summary, generate_html_report, generate_latex_table,
    plot_accuracy_vs_rank, plot_radar_chart
)

# Auto-register all built‑in plots
register_default_plots()

__all__ = [
    "BaseVisualizer", "VISUALIZER_REGISTRY", "Registry",
    "PLOT_REGISTRY",
    "set_plot_style", "save_figure", "get_color_palette", "create_custom_colormap",
    "generate_csv_summary", "generate_html_report", "generate_latex_table",
    "plot_accuracy_vs_rank", "plot_radar_chart",
]