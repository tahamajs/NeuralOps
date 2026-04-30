# framework/analysis/__init__.py
"""
Analysis and visualization tools for SVD compression framework.

This module provides:
- Model comparison across different ranks and tasks
- Layer-wise analysis of hidden states and attention
- Comprehensive visualization generation
- Report generation (HTML, CSV, JSON)
- Metrics computation and aggregation

Example:
    >>> from framework.analysis import ModelComparison, LayerwiseAnalyzer
    >>> 
    >>> # Compare models
    >>> comparator = ModelComparison(results_dirs=['./results/run1', './results/run2'])
    >>> comparator.load_all_results()
    >>> comparator.plot_accuracy_vs_rank(task='cola')
    >>> 
    >>> # Analyze layer-wise properties
    >>> analyzer = LayerwiseAnalyzer(model, tokenizer)
    >>> results = analyzer.analyze("Sample text")
"""

from .model_comparison import ModelComparison, quick_compare
from .layerwise_analyzer import LayerwiseAnalyzer
from .visualization import (
    create_comprehensive_plots,
    create_individual_plots,
    create_tradeoff_plot,
    plot_training_curves,
    plot_confusion_matrix,
    plot_attention_heatmap,
)
from .report_generator import (
    generate_html_report,
    generate_csv_summary,
    generate_latex_table,
)
from .metrics import (
    compute_metrics,
    aggregate_results,
    compute_accuracy_drop,
    compute_efficiency_score,
)
from .utils import (
    load_results_from_directory,
    merge_results,
    filter_results,
    get_best_configurations,
)

__all__ = [
    # Model comparison
    "ModelComparison",
    "quick_compare",
    
    # Layer analysis
    "LayerwiseAnalyzer",
    
    # Visualization
    "create_comprehensive_plots",
    "create_individual_plots",
    "create_tradeoff_plot",
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_attention_heatmap",
    
    # Reports
    "generate_html_report",
    "generate_csv_summary",
    "generate_latex_table",
    
    # Metrics
    "compute_metrics",
    "aggregate_results",
    "compute_accuracy_drop",
    "compute_efficiency_score",
    
    # Utilities
    "load_results_from_directory",
    "merge_results",
    "filter_results",
    "get_best_configurations",
]

__version__ = "1.0.0"