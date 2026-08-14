"""
ResultsSaver – central orchestrator for result persistence.
Integrates with framework.analysis, framework.tracking, and all backends.
"""

import json
import logging
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config import SaveConfig

logger = logging.getLogger(__name__)


class ResultsSaver:
    """
    Usage:
        saver = ResultsSaver(run_id, base_dir, save_config_dict)
        saver.save_all(model, config, history, final_metrics, tokenizer, input_example)
        saver.push_all()
    """

    def __init__(
        self, run_id: str, base_dir: Path, config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.run_id = run_id
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / run_id
        self.cfg = SaveConfig.from_dict(config or {})
        self._init_dirs()
        self.saved_paths: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Directory setup
    # ------------------------------------------------------------------
    def _init_dirs(self) -> None:
        """ init dirs."""
        dirs = [self.run_dir]
        if self.cfg.save_logs:
            dirs.append(self.run_dir / "logs")
        if self.cfg.save_plots:
            dirs.append(self.run_dir / "plots")
        if (
            self.cfg.save_artifacts
            or self.cfg.save_model
            or self.cfg.model_export_formats
        ):
            dirs.append(self.run_dir / "artifacts")
        if self.cfg.save_checkpoints:
            dirs.append(self.run_dir / "checkpoints")
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def should_save_heavy(self, final_metrics: Dict[str, Any]) -> bool:
        """Should save heavy."""
        threshold = self.cfg.auto_save_if_better_than
        if threshold is None:
            return True
        metric = final_metrics.get("eval_accuracy", 0)
        return metric >= threshold

    # ------------------------------------------------------------------
    # Public convenience methods
    # ------------------------------------------------------------------
    def save_metrics(self, metrics: Dict[str, Any]) -> str:
        """Save a dictionary of evaluation/training metrics."""
        self._save_final_metrics(metrics)
        path = str(self.run_dir / "artifacts" / "final_metrics.json")
        self.saved_paths["metrics"] = path
        return path

    def save_config(self, config: Any) -> str:
        """Save experiment configuration dictionary or object."""
        self._save_config(config)
        path = str(self.run_dir / "artifacts" / "config.json")
        self.saved_paths["config"] = path
        return path

    def save_model_weights(self, model: Any, filename: str = "model.pt") -> str:
        """Save PyTorch model state dict or checkpoint."""
        import torch
        art_dir = self.run_dir / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)
        dest = art_dir / filename
        if hasattr(model, "state_dict"):
            torch.save(model.state_dict(), dest)
        elif hasattr(model, "save_pretrained"):
            model.save_pretrained(str(art_dir / "model"))
            return str(art_dir / "model")
        else:
            torch.save(model, dest)
        self.saved_paths["model_weights"] = str(dest)
        return str(dest)

    # ------------------------------------------------------------------
    # Main save entry point
    # ------------------------------------------------------------------
    def save_all(
        self, model, config, history, final_metrics, tokenizer=None, input_example=None
    ) -> None:
        """Save everything selected in the config."""

        self._save_config(config)
        self._save_history(history)
        self._save_final_metrics(final_metrics)
        self._save_metrics_log(history)
        if self.cfg.save_model and self.should_save_heavy(final_metrics):
            self._save_model(model, tokenizer)
        if self.cfg.save_checkpoints:
            self._save_checkpoint(model, "final")
        if self.cfg.save_plots:
            self._generate_plots(history, final_metrics)
        if self.cfg.save_model and self.should_save_heavy(final_metrics):
            for fmt in self.cfg.model_export_formats:
                self._export_format(fmt, model, input_example)
        self._save_archive()
        self._save_model_card(config, final_metrics)

        # Automatically track to experiment trackers (wandb, mlflow, etc.)
        if self.cfg.experiment_trackers:
            # Delegate to the unified tracking module if available, else use legacy backends
            try:
                from framework.tracking import ExperimentTracker

                tracker = ExperimentTracker(
                    backend=self.cfg.experiment_trackers[0],  # supports one primary
                    log_dir=str(self.run_dir / "tracking"),
                    experiment_name=self.run_id,
                )
                for k, v in final_metrics.items():
                    tracker.log_metric(k, v)
                tracker.log_params(self._flatten_config(config))
                tracker.finalize()
            except ImportError:
                self.track_all(final_metrics, self._flatten_config(config))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _write_json(self, rel_path: str, obj) -> None:
        """ write json."""
        def _default_serializer(o):
            if hasattr(o, "to_dict") and callable(o.to_dict):
                return o.to_dict()
            if hasattr(o, "__dict__"):
                return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
            if isinstance(o, (datetime, Path)):
                return str(o)
            if hasattr(o, "value"):
                return o.value
            return str(o)

        fp = self.run_dir / rel_path
        fp.parent.mkdir(exist_ok=True)
        with open(fp, "w") as f:
            json.dump(obj, f, indent=2, default=_default_serializer)

    def _write_text(self, rel_path: str, text: str) -> None:
        """ write text."""
        fp = self.run_dir / rel_path
        fp.parent.mkdir(exist_ok=True)
        with open(fp, "w") as f:
            f.write(text)

    def _flatten_config(self, config) -> Dict[str, Any]:
        """ flatten config."""
        if hasattr(config, "__dict__"):
            return vars(config)
        return config

    def _save_config(self, config) -> None:
        """ save config."""
        self._write_json("artifacts/config.json", self._flatten_config(config))

    def _save_history(self, history) -> None:
        """ save history."""
        self._write_json("artifacts/history.json", history)

    def _save_final_metrics(self, metrics) -> None:
        """ save final metrics."""
        self._write_json("artifacts/final_metrics.json", metrics)

    def _save_metrics_log(self, history) -> None:
        """ save metrics log."""
        with open(self.run_dir / "logs" / "metrics.jsonl", "w") as f:
            for entry in history:
                f.write(json.dumps(entry) + "\n")

    def _save_model(self, model, tokenizer=None) -> None:
        """ save model."""
        p = self.run_dir / "artifacts" / "model"
        p.mkdir(exist_ok=True)
        model.save_pretrained(str(p))
        if tokenizer:
            tokenizer.save_pretrained(str(p))
        if self.cfg.save_optimizer:
            import torch

            torch.save(model.state_dict(), p / "optimizer.pt")

    def _save_checkpoint(self, model, name="best") -> None:
        """ save checkpoint."""
        p = self.run_dir / "checkpoints" / name
        p.mkdir(exist_ok=True)
        model.save_pretrained(str(p))

    def _save_archive(self) -> None:
        """ save archive safely."""
        try:
            with tarfile.open(self.run_dir / "results.tar.gz", "w:gz") as tar:
                for sub in ["artifacts", "logs", "plots"]:
                    sp = self.run_dir / sub
                    if sp.exists():
                        tar.add(sp, arcname=sub)
        except Exception as e:
            logger.warning(f"Archive creation skipped due to warning/error: {e}")

    # ------------------------------------------------------------------
    # Plots – delegates to framework.analysis.plotting
    # ------------------------------------------------------------------
    def _generate_plots(self, history, final_metrics) -> None:
        """ generate plots."""
        try:
            import matplotlib.pyplot as plt
            from framework.analysis.plotting import (
                plot_accuracy_curves,
                plot_confusion_matrix,
                plot_gpu_memory,
                plot_gradient_flow,
                plot_loss_curves,
                plot_lr_schedule,
                plot_radar_chart,
                plot_rank_allocation,
                plot_training_time,
            )

            opts = self.cfg.plot_options
            dpi = opts.get("dpi", 150)

            def save_fig(name, fig) -> None:
                """Persist fig to disk."""
                fig.savefig(
                    self.run_dir / "plots" / f"{name}.png", dpi=dpi, bbox_inches="tight"
                )
                plt.close(fig)

            if opts.get("loss_curve", True):
                try:
                    plot_loss_curve(history, final_metrics, self.cfg, self.run_dir / "plots", opts)
                except Exception as e:
                    logger.debug(f"plot_loss_curve error: {e}")

            if opts.get("accuracy_curve", True):
                try:
                    plot_accuracy_curve(history, final_metrics, self.cfg, self.run_dir / "plots", opts)
                except Exception as e:
                    logger.debug(f"plot_accuracy_curve error: {e}")

            if opts.get("lr_schedule", True):
                try:
                    plot_lr_schedule(history, final_metrics, self.cfg, self.run_dir / "plots", opts)
                except Exception as e:
                    logger.debug(f"plot_lr_schedule error: {e}")

            if opts.get("training_time", True):
                try:
                    plot_training_time(history, final_metrics, self.cfg, self.run_dir / "plots", opts)
                except Exception as e:
                    logger.debug(f"plot_training_time error: {e}")

            if opts.get("confusion_matrix", False):
                preds_path = self.run_dir / "artifacts" / "eval_predictions.json"
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
                fig = plot_radar_chart(
                    final_metrics, figsize=tuple(opts.get("figsize", [8, 8]))
                )
                save_fig("radar_chart", fig)

            if opts.get("rank_allocation", False):
                rank_path = self.run_dir / "artifacts" / "rank_allocation.json"
                if rank_path.exists():
                    with open(rank_path) as f:
                        rank_data = json.load(f)
                    fig = plot_rank_allocation(
                        rank_data, figsize=tuple(opts.get("figsize", [10, 4]))
                    )
                    save_fig("rank_allocation", fig)

            if opts.get("attention_heatmap", False):
                logger.info(
                    "Attention heatmap requested but model not available in saver context; "
                    "use framework.callbacks.AttentionHeatmapCallback during training."
                )

        except Exception:
            logger.exception("Plot generation failed")

    # ------------------------------------------------------------------
    # Model card
    # ------------------------------------------------------------------
    def _save_model_card(self, config, metrics) -> None:
        """ save model card."""
        cfg_dict = (
            self._flatten_config(config) if hasattr(config, "__dict__") else config
        )
        model_name = cfg_dict.get("model_name", "Model")
        task_name = cfg_dict.get("task_name", "Task")
        acc = metrics.get("eval_accuracy", "N/A")
        loss = metrics.get("eval_loss", "N/A")
        card = f"""# {model_name} – {task_name}

**Accuracy:** {acc}
**Loss:** {loss}

**Run ID:** {self.run_id}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Compression:** {cfg_dict.get('attention_type', 'full')} (rank={cfg_dict.get('rank', 'N/A')})
"""
        self._write_text("artifacts/model_card.md", card)

    # ------------------------------------------------------------------
    # Export formats
    # ------------------------------------------------------------------
    def _export_format(self, fmt: str, model, input_example) -> None:
        """ export format."""
        try:
            from .backends.exports import EXPORTERS

            exporter = EXPORTERS.get(fmt)
            if exporter:
                exporter(model, input_example, str(self.run_dir / "artifacts"))
        except Exception:
            logger.exception(f"Export {fmt} failed")

    # ------------------------------------------------------------------
    # Push to storage backends
    # ------------------------------------------------------------------
    def push_all(self) -> None:
        """Push all."""
        for be in self.cfg.storage_backends:
            self._push_backend(be)

    def _push_backend(self, name: str) -> None:
        """ push backend."""
        try:
            if name == "local":
                return
            mod = __import__(
                f"framework.results_saver.backends.{name}", fromlist=["push"]
            )
            mod.push(self.run_dir, self.cfg)
        except ImportError:
            logger.warning(f"Backend '{name}' not available, skipping")
        except Exception:
            logger.exception(f"Push to {name} failed")

    # ------------------------------------------------------------------
    # Git / DVC
    # ------------------------------------------------------------------
    def git_commit_and_push(self) -> None:
        """Git commit and push."""
        if not self.cfg.git_options.get("commit_results"):
            return
        from .backends.git import git_commit_and_push

        git_commit_and_push(self.run_dir, self.cfg)

    def dvc_track(self) -> None:
        """Dvc track."""
        if not self.cfg.dvc_options.get("track_model"):
            return
        from .backends.dvc import dvc_track

        dvc_track(self.run_dir, self.cfg)

    # ------------------------------------------------------------------
    # Experiment trackers (legacy backends – prefer framework.tracking)
    # ------------------------------------------------------------------
    def track_all(self, metrics: Dict[str, Any], params: Dict[str, Any]) -> None:
        """Track all."""
        for t in self.cfg.experiment_trackers:
            try:
                mod = __import__(
                    f"framework.results_saver.backends.{t}_tracker", fromlist=["track"]
                )
                mod.track(self.run_id, self.run_dir, metrics, params, self.cfg)
            except ImportError:
                logger.warning(f"Tracker '{t}' not available, skipping")
            except Exception:
                logger.exception(f"Tracking to {t} failed")

    # ------------------------------------------------------------------
    # Model registries
    # ------------------------------------------------------------------
    def push_to_registries(self) -> None:
        """Push to registries."""
        for r in self.cfg.model_registries:
            try:
                mod = __import__(
                    f"framework.results_saver.backends.{r}", fromlist=["push"]
                )
                mod.push(self.run_id, self.run_dir, self.cfg)
            except ImportError:
                logger.warning(f"Registry '{r}' not available")
            except Exception:
                logger.exception(f"Push to registry {r} failed")

    # ------------------------------------------------------------------
    # GitHub Release
    # ------------------------------------------------------------------
    def create_github_release(self, metrics: Dict[str, Any]) -> Optional[str]:
        """Build and return github release."""
        if not self.cfg.release_options.get("create_github_release"):
            return None
        from .backends.github_release import create_release

        return create_release(self.run_id, self.run_dir, metrics, self.cfg)

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------
    def notify(self, success: bool, metrics: Dict[str, Any]) -> None:
        """Notify."""
        from .backends.notification import notify

        notify(self.run_id, success, metrics, self.cfg)
