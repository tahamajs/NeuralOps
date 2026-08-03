"""
Analysis tools: plotting, metrics, layerwise analysis, model comparison, reporting.
"""

from .layerwise_analyzer import LayerwiseAnalyzer, estimate_intrinsic_dimension
from .metrics import MetricsTracker, aggregate_results, compute_metrics
from .model_comparison import ModelComparison, quick_compare
from .plotting import (
    create_comprehensive_plots,
    plot_accuracy_curve,
    plot_attention_heatmap,
    plot_confusion_matrix,
    plot_gpu_memory,
    plot_gradient_flow,
    plot_loss_curve,
    plot_lr_schedule,
    plot_parameter_distribution,
    plot_pareto_front,
    plot_radar_chart,
    plot_rank_allocation,
    plot_tradeoff,
    plot_training_history,
    plot_training_time,
    set_style,
)
from .report_generator import (
    generate_full_report,
    generate_html_report,
    generate_interactive_report,
    generate_latex_table,
)
from .utils import (
    correlation_matrix,
    detect_least_compressible_layers,
    filter_results,
    get_best_configurations,
    statistical_tests,
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
