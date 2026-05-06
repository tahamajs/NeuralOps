from .plotting import *
from .metrics import MetricsTracker, compute_metrics, aggregate_results
from .layerwise_analyzer import LayerwiseAnalyzer, estimate_intrinsic_dimension
from .model_comparison import ModelComparison, quick_compare
from .report_generator import (
    generate_html_report, generate_latex_table, generate_full_report,
    generate_interactive_report,
)
from .callbacks import LoggingCallback
from .utils import (
    detect_least_compressible_layers, filter_results, get_best_configurations,
    correlation_matrix, statistical_tests,
)

from .comparison import (
    ComparisonEngine,
    ComparisonResult,
    generate_leaderboard,
    anova_test,
    pairwise_t_tests,
    mann_whitney_test,
    nemenyi_test,
    dunn_test,
    plot_critical_difference_diagram,
    plot_pairwise_comparison_heatmap,
)