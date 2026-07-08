"""
Pluggable plot types for the ResultsSaver.
Each function receives (history, metrics, config, output_dir, opts) and returns a list of saved file paths.
"""

import logging

from .base import Registry

PLOT_REGISTRY = Registry("plots")
logger = logging.getLogger(__name__)


def register_default_plots():
    """Call this once to register all built‑in plot types."""
    from .plots.accuracy import plot_accuracy_curve
    from .plots.activation_dist import plot_activation_distribution
    from .plots.attention import plot_attention_heatmap
    from .plots.confusion import plot_confusion_matrix
    from .plots.embedding import plot_embedding
    from .plots.gpu_memory import plot_gpu_memory
    from .plots.gradient import plot_gradient_flow
    from .plots.hyperparam_importance import plot_hyperparam_importance
    from .plots.learning_curves import plot_learning_curve
    from .plots.learning_rate import plot_lr_schedule
    from .plots.loss import plot_loss_curve
    from .plots.parallel_coords import plot_parallel_coords
    from .plots.parameter_count import plot_parameter_distribution
    from .plots.precision_recall_by_class import plot_precision_recall_by_class
    from .plots.radar_chart import plot_radar_chart
    from .plots.rank_allocation import plot_rank_allocation
    from .plots.residuals import plot_residuals
    from .plots.robustness import plot_robustness
    from .plots.roc_pr_curve import plot_roc_pr_curve
    from .plots.training_curves import plot_combined_training_curves
    from .plots.training_time import plot_training_time

    PLOT_REGISTRY.register("roc_pr_curve")(plot_roc_pr_curve)
    PLOT_REGISTRY.register("learning_curve")(plot_learning_curve)
    PLOT_REGISTRY.register("embedding")(plot_embedding)
    PLOT_REGISTRY.register("residuals")(plot_residuals)
    PLOT_REGISTRY.register("parallel_coords")(plot_parallel_coords)
    PLOT_REGISTRY.register("hyperparam_importance")(plot_hyperparam_importance)
    PLOT_REGISTRY.register("robustness")(plot_robustness)
    PLOT_REGISTRY.register("activation_dist")(plot_activation_distribution)
    PLOT_REGISTRY.register("precision_recall_by_class")(plot_precision_recall_by_class)

    PLOT_REGISTRY.register("loss_curve")(plot_loss_curve)
    PLOT_REGISTRY.register("accuracy_curve")(plot_accuracy_curve)
    PLOT_REGISTRY.register("lr_schedule")(plot_lr_schedule)
    PLOT_REGISTRY.register("training_time")(plot_training_time)
    PLOT_REGISTRY.register("combined_curves")(plot_combined_training_curves)
    PLOT_REGISTRY.register("confusion_matrix")(plot_confusion_matrix)
    PLOT_REGISTRY.register("gradient_flow")(plot_gradient_flow)
    PLOT_REGISTRY.register("gpu_memory")(plot_gpu_memory)
    PLOT_REGISTRY.register("parameter_count")(plot_parameter_distribution)
    PLOT_REGISTRY.register("rank_allocation")(plot_rank_allocation)
    PLOT_REGISTRY.register("radar_chart")(plot_radar_chart)
    PLOT_REGISTRY.register("attention_heatmap")(plot_attention_heatmap)
    logger.info("Registered %d plot types", len(PLOT_REGISTRY))
