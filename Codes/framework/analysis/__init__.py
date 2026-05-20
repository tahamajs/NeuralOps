"""
Analysis tools: plotting, metrics, layerwise analysis, model comparison, reporting.
"""

from .plotting import (
    plot_loss_curve, plot_accuracy_curve, plot_lr_schedule, plot_training_time,
    plot_confusion_matrix, plot_gradient_flow, plot_gpu_memory,
    plot_parameter_distribution, plot_rank_allocation, plot_radar_chart,
    plot_attention_heatmap, create_comprehensive_plots, plot_tradeoff,
    plot_pareto_front, plot_training_history, set_style,
)
from .metrics import MetricsTracker, compute_metrics, aggregate_results
from .layerwise_analyzer import LayerwiseAnalyzer, estimate_intrinsic_dimension
from .model_comparison import ModelComparison, quick_compare
from .report_generator import (
    generate_html_report, generate_latex_table, generate_full_report,
    generate_interactive_report,
)
from .utils import (
    detect_least_compressible_layers, filter_results, get_best_configurations,
    correlation_matrix, statistical_tests,
)

__all__ = [
    "plot_loss_curve",
    "plot_accuracy_curve",
    "plot_lr_schedule",
    "plot_training_time",
    "plot_confusion_matrix",
    "plot_gradient_flow",
    "plot_gpu_memory",
    "plot_parameter_distribution",
    "plot_rank_allocation",
    "plot_radar_chart",
    "plot_attention_heatmap",
    "create_comprehensive_plots",
    "plot_tradeoff",
    "plot_pareto_front",
    "plot_training_history",
    "set_style",
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
    "generate_interactive_report",
    "detect_least_compressible_layers",
    "filter_results",
    "get_best_configurations",
    "correlation_matrix",
    "statistical_tests",
]