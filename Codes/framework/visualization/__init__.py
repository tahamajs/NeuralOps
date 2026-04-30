# framework/visualization/__init__.py
"""
Visualization and reporting tools for SVD-Lin framework.

This module provides:
- Attention visualizations (heatmaps, entropy, difference maps)
- Training curve plotting
- Model comparison plots
- Interactive HTML reports with Plotly
- CSV and LaTeX report generation

Example:
    >>> from framework.visualization import SVDAttentionVisualizer, AdvancedReportGenerator
    >>> 
    >>> # Create attention visualizations
    >>> vis = SVDAttentionVisualizer(output_dir="./vis")
    >>> vis.plot_attention_heatmap(attention_matrix, tokens, layer=0, head=0, rank=64)
    >>> 
    >>> # Generate interactive report
    >>> report_gen = AdvancedReportGenerator(output_dir="./report")
    >>> report_gen.generate_full_report(results, df, profiler_report, layer_analysis)
"""

from .base_visualizer import BaseVisualizer
from .attention_visualizer import SVDAttentionVisualizer
from .advanced_report import AdvancedReportGenerator
from .report_generator import generate_csv_summary, generate_html_report, generate_latex_table
from .training_plots import plot_training_curves, plot_learning_rate_schedule
from .comparison_plots import (
    plot_accuracy_vs_rank,
    plot_model_comparison,
    plot_rank_comparison,
    plot_accuracy_drop_distribution,
    plot_radar_chart,
)
from .utils import set_plot_style, save_figure, get_color_palette

__all__ = [
    # Base
    "BaseVisualizer",
    
    # Attention
    "SVDAttentionVisualizer",
    
    # Reports
    "AdvancedReportGenerator",
    "generate_csv_summary",
    "generate_html_report",
    "generate_latex_table",
    
    # Training
    "plot_training_curves",
    "plot_learning_rate_schedule",
    
    # Comparison
    "plot_accuracy_vs_rank",
    "plot_model_comparison",
    "plot_rank_comparison",
    "plot_accuracy_drop_distribution",
    "plot_radar_chart",
    
    # Utils
    "set_plot_style",
    "save_figure",
    "get_color_palette",
]

__version__ = "1.0.0"