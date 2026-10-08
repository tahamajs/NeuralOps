"""Plotting delegator for experiment history and metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_plots_for_run(
    run_dir: Path,
    cfg: Any,
    history: list[dict[str, Any]] | dict[str, Any],
    final_metrics: dict[str, Any],
) -> None:
    """Generate conference-grade plots delegating to neuralops.analysis.plotting.

    Args:
        run_dir: Base directory of the experiment run.
        cfg: SaveConfig instance containing plot options.
        history: Step or epoch training history data.
        final_metrics: Final evaluation metric mapping.
    """
    try:
        import matplotlib.pyplot as plt

        from neuralops.analysis.plotting import (
            plot_accuracy_curves,
            plot_confusion_matrix,
            plot_loss_curves,
            plot_lr_schedule,
            plot_radar_chart,
            plot_rank_allocation,
            plot_training_time,
        )

        opts = cfg.plot_options
        dpi = opts.get("dpi", 150)
        plots_dir = run_dir / "plots"

        def save_fig(name: str, fig: Any) -> None:
            fig.savefig(plots_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        if opts.get("loss_curve", True):
            try:
                plot_loss_curves(history, final_metrics, cfg, plots_dir, opts)
            except Exception as e:
                logger.debug(f"plot_loss_curves error: {e}")

        if opts.get("accuracy_curve", True):
            try:
                plot_accuracy_curves(history, final_metrics, cfg, plots_dir, opts)
            except Exception as e:
                logger.debug(f"plot_accuracy_curves error: {e}")

        if opts.get("lr_schedule", True):
            try:
                plot_lr_schedule(history, final_metrics, cfg, plots_dir, opts)
            except Exception as e:
                logger.debug(f"plot_lr_schedule error: {e}")

        if opts.get("training_time", True):
            try:
                plot_training_time(history, final_metrics, cfg, plots_dir, opts)
            except Exception as e:
                logger.debug(f"plot_training_time error: {e}")

        if opts.get("confusion_matrix", False):
            preds_path = run_dir / "artifacts" / "eval_predictions.json"
            if preds_path.exists():
                with open(preds_path) as f:
                    data = json.load(f)
                fig = plot_confusion_matrix(
                    data["labels"],
                    data["predictions"],
                    class_names=opts.get("class_names"),
                    figsize=tuple(opts.get("figsize", [10, 8])),
                )
                save_fig("confusion_matrix", fig)

        if opts.get("radar_chart", False):
            fig = plot_radar_chart(final_metrics, figsize=tuple(opts.get("figsize", [8, 8])))
            save_fig("radar_chart", fig)

        if opts.get("rank_allocation", False):
            rank_path = run_dir / "artifacts" / "rank_allocation.json"
            if rank_path.exists():
                with open(rank_path) as f:
                    rank_data = json.load(f)
                fig = plot_rank_allocation(rank_data, figsize=tuple(opts.get("figsize", [10, 4])))
                save_fig("rank_allocation", fig)

        if opts.get("attention_heatmap", False):
            logger.info(
                "Attention heatmap requested but model not available in saver context; "
                "use framework.callbacks.AttentionHeatmapCallback during training."
            )

    except Exception:
        logger.exception("Plot generation failed")
