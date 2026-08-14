from .base import VISUALIZER_REGISTRY, BaseVisualizer, Registry
from .benchmark_visualizer import (
    plot_attention_heatmap,
    plot_dataset_radar_chart,
    plot_kv_cache_scaling,
    plot_layerwise_sparsity_rank,
    plot_pareto_frontier,
    plot_singular_value_decay,
)
from .dashboard_generator import generate_dashboard_html, generate_summary_json
from .plot_registry import PLOT_REGISTRY, register_default_plots
from .plots import *  # convenience so all plots are importable
from .report_generator import (
    generate_csv_summary,
    generate_html_report,
    generate_latex_table,
    plot_accuracy_vs_rank,
    plot_radar_chart,
)
from .utils import (
    create_custom_colormap,
    get_color_palette,
    save_figure,
    set_plot_style,
)

# Auto-register all built‑in plots
register_default_plots()

__all__ = [
    "BaseVisualizer",
    "VISUALIZER_REGISTRY",
    "Registry",
    "PLOT_REGISTRY",
    "set_plot_style",
    "save_figure",
    "get_color_palette",
    "create_custom_colormap",
    "generate_csv_summary",
    "generate_html_report",
    "generate_latex_table",
    "plot_accuracy_vs_rank",
    "plot_radar_chart",
    "plot_attention_heatmap",
    "plot_singular_value_decay",
    "plot_kv_cache_scaling",
    "plot_pareto_frontier",
    "plot_dataset_radar_chart",
    "plot_layerwise_sparsity_rank",
    "generate_dashboard_html",
    "generate_summary_json",
]

