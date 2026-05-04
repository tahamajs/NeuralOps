# framework/analysis/__init__.py
from .plotting import (
    create_comprehensive_plots,
    plot_tradeoff,
    plot_pareto_front,
    plot_rank_allocation,
    plot_training_history,
)
from .metrics import MetricsTracker, compute_metrics, aggregate_results
from .layerwise_analyzer import LayerwiseAnalyzer, estimate_intrinsic_dimension
from .model_comparison import ModelComparison, quick_compare
from .report_generator import generate_html_report, generate_latex_table, generate_full_report
from .callbacks import LoggingCallback
from .utils import detect_least_compressible_layers, filter_results, get_best_configurations

__all__ = [
    "create_comprehensive_plots",
    "plot_tradeoff",
    "plot_pareto_front",
    "plot_rank_allocation",
    "plot_training_history",
    "MetricsTracker",
    "compute_metrics",
    "aggregate_results",
    "LayerwiseAnalyzer",
    "estimate_intrinsic_dimension",
    "ModelComparison",
    "quick_compare",
    "generate_html_report",
    "generate_latex_table",
    "generate_full_report",
    "LoggingCallback",
    "detect_least_compressible_layers",
    "filter_results",
    "get_best_configurations",
]