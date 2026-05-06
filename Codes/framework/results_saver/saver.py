import os, json, logging, tarfile, tempfile, shutil
from pathlib import Path
from datetime import datetime
from typing import Type, Optional, Dict, Any

from .config import SaveConfig

logger = logging.getLogger(__name__)

class ResultsSaver:
    """
    Central orchestrator for result persistence.
    Usage:
        saver = ResultsSaver(run_id, base_dir, save_config_dict)
        saver.save_all(model, config, history, metrics, tokenizer, input_example)
        saver.push_all()
    """

    def __init__(self, run_id: str, base_dir: Path, config: Optional[Dict[str, Any]] = None):
        self.run_id = run_id
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / str(run_id)
        self.cfg = SaveConfig.from_dict(config or {})
        self._init_dirs()
        self.saved_paths: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Directory setup
    # ------------------------------------------------------------------
    def _init_dirs(self):
        dirs = [self.run_dir]
        if self.cfg.save_logs:      dirs.append(self.run_dir / "logs")
        if self.cfg.save_plots:     dirs.append(self.run_dir / "plots")
        if self.cfg.save_artifacts or self.cfg.save_model or self.cfg.model_export_formats:
            dirs.append(self.run_dir / "artifacts")
        if self.cfg.save_checkpoints:  dirs.append(self.run_dir / "checkpoints")
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def should_save_heavy(self, final_metrics: Dict[str, Any]) -> bool:
        threshold = self.cfg.auto_save_if_better_than
        if threshold is None:
            return True
        metric = final_metrics.get("eval_accuracy", 0)
        return metric >= threshold

    # ------------------------------------------------------------------
    # Core save operations
    # ------------------------------------------------------------------
    def save_all(self, model, config, history, final_metrics,
                 tokenizer=None, input_example=None) -> None:
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
            self._generate_plots(history)
        if self.cfg.save_model and self.should_save_heavy(final_metrics):
            for fmt in self.cfg.model_export_formats:
                self._export_format(fmt, model, input_example)
        self._save_archive()
        self._save_model_card(config, final_metrics)

    def _write_json(self, rel_path: str, obj):
        fp = self.run_dir / rel_path; fp.parent.mkdir(exist_ok=True)
        with open(fp, 'w') as f: json.dump(obj, f, indent=2)

    def _write_text(self, rel_path: str, text: str):
        fp = self.run_dir / rel_path; fp.parent.mkdir(exist_ok=True)
        with open(fp, 'w') as f: f.write(text)

    def _save_config(self, config):
        d = vars(config) if hasattr(config, '__dict__') else config
        self._write_json("artifacts/config.json", d)

    def _save_history(self, history):
        self._write_json("artifacts/history.json", history)

    def _save_final_metrics(self, metrics):
        self._write_json("artifacts/final_metrics.json", metrics)

    def _save_metrics_log(self, history):
        with open(self.run_dir / "logs" / "metrics.jsonl", 'w') as f:
            for entry in history:
                f.write(json.dumps(entry) + "\n")

    def _save_model(self, model, tokenizer=None):
        p = self.run_dir / "artifacts" / "model"; p.mkdir(exist_ok=True)
        model.save_pretrained(str(p))
        if tokenizer: tokenizer.save_pretrained(str(p))
        if self.cfg.save_optimizer:
            import torch
            torch.save(model.state_dict(), p / "optimizer.pt")

    def _save_checkpoint(self, model, name="best"):
        p = self.run_dir / "checkpoints" / name; p.mkdir(exist_ok=True)
        model.save_pretrained(str(p))

    def _save_archive(self):
        with tarfile.open(self.run_dir / "results.tar.gz", "w:gz") as tar:
            for sub in ["artifacts", "logs", "plots"]:
                sp = self.run_dir / sub
                if sp.exists(): tar.add(sp, arcname=sub)

    def _generate_plots(self, history):
        try:
            from framework.analysis.plotting import create_comprehensive_plots
            create_comprehensive_plots(history, str(self.run_dir / "plots"))
        except Exception:
            logger.exception("Plot generation failed")

    def _save_model_card(self, config, metrics):
        card = f"# {config.get('model_name','Model')} – {config.get('task_name','Task')}\n\n"
        card += f"**Accuracy:** {metrics.get('eval_accuracy','N/A')}\n"
        card += f"**Loss:** {metrics.get('eval_loss','N/A')}\n\n"
        card += f"**Run ID:** {self.run_id}\n**Date:** {datetime.now()}\n"
        self._write_text("artifacts/model_card.md", card)

    def _export_format(self, fmt: str, model, input_example):
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
        for be in self.cfg.storage_backends:
            self._push_backend(be)

    def _push_backend(self, name: str):
        try:
            if name == "local": return
            mod = __import__(f".backends.{name}", fromlist=["push"], package=__package__)
            mod.push(self.run_dir, self.cfg)
        except ImportError:
            logger.warning(f"Backend '{name}' not available, skipping")
        except Exception:
            logger.exception(f"Push to {name} failed")

    # ------------------------------------------------------------------
    # Git / DVC
    # ------------------------------------------------------------------
    def git_commit_and_push(self) -> None:
        if not self.cfg.git_options.get("commit_results"): return
        from .backends.git import git_commit_and_push
        git_commit_and_push(self.run_dir, self.cfg)

    def dvc_track(self) -> None:
        if not self.cfg.dvc_options.get("track_model"): return
        from .backends.dvc import dvc_track
        dvc_track(self.run_dir, self.cfg)

    # ------------------------------------------------------------------
    # Experiment trackers
    # ------------------------------------------------------------------
    def track_all(self, metrics: Dict[str, Any], params: Dict[str, Any]) -> None:
        for t in self.cfg.experiment_trackers:
            try:
                mod = __import__(f".backends.{t}", fromlist=["track"], package=__package__)
                mod.track(self.run_id, self.run_dir, metrics, params, self.cfg)
            except ImportError:
                logger.warning(f"Tracker '{t}' not available, skipping")
            except Exception:
                logger.exception(f"Tracking to {t} failed")

    # ------------------------------------------------------------------
    # Model registries
    # ------------------------------------------------------------------
    def push_to_registries(self) -> None:
        for r in self.cfg.model_registries:
            try:
                mod = __import__(f".backends.{r}", fromlist=["push"], package=__package__)
                mod.push(self.run_id, self.run_dir, self.cfg)
            except ImportError:
                logger.warning(f"Registry '{r}' not available")
            except Exception:
                logger.exception(f"Push to registry {r} failed")

    # ------------------------------------------------------------------
    # GitHub Release
    # ------------------------------------------------------------------
    def create_github_release(self, metrics: Dict[str, Any]) -> Optional[str]:
        if not self.cfg.release_options.get("create_github_release"): return None
        from .backends.github_release import create_release
        return create_release(self.run_id, self.run_dir, metrics, self.cfg)

    # ------------------------------------------------------------------
    # Model version (DB)
    # ------------------------------------------------------------------
    def create_version(self, run, metrics: Dict[str, Any]) -> None:
        if not self.cfg.create_model_version: return
        from model_versioning.services import ModelVersionService
        p = self.run_dir / "artifacts" / "model" / "pytorch_model.bin"
        if p.exists():
            import torch
            state = torch.load(p, map_location='cpu')
            ModelVersionService.create_version(run=run, model_state=state,
                                               metrics=metrics, artifact_base=str(self.base_dir))

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------
    def notify(self, success: bool, metrics: Dict[str, Any]) -> None:
        from .backends.notification import notify
        notify(self.run_id, success, metrics, self.cfg)

# framework/results_saver/saver.py (inside ResultsSaver class)

    def _generate_plots(self, history):
        if not self.cfg.save_plots:
            return
        opts = self.cfg.plot_options   # guaranteed to exist as a dict

        # Global style settings
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-darkgrid')
        figsize = tuple(opts.get("figsize", [10, 6]))
        dpi = opts.get("dpi", 150)

        # Helper to save a single figure
        def _save_figure(name, fig):
            fig.savefig(self.run_dir / "plots" / f"{name}.png", dpi=dpi, bbox_inches='tight')
            plt.close(fig)

        # ----- Loss curves -----
        if opts.get("loss_curve", True):
            from framework.analysis.plotting import plot_loss_curves
            fig = plot_loss_curves(history, smooth=opts.get("loss_smooth", 0.6), figsize=figsize)
            _save_figure("loss_curve", fig)

        # ----- Accuracy curves -----
        if opts.get("accuracy_curve", True):
            from framework.analysis.plotting import plot_accuracy_curves
            fig = plot_accuracy_curves(history, figsize=figsize)
            _save_figure("accuracy_curve", fig)

        # ----- Confusion matrix -----
        if opts.get("confusion_matrix", False):
            from framework.analysis.plotting import plot_confusion_matrix
            # Requires evaluation predictions + labels – must be passed separately
            # We store them in the run directory if available
            preds_path = self.run_dir / "artifacts" / "eval_predictions.json"
            if preds_path.exists():
                import json
                with open(preds_path) as f:
                    data = json.load(f)
                fig = plot_confusion_matrix(data["labels"], data["predictions"],
                                            class_names=opts.get("class_names"),
                                            figsize=figsize)
                _save_figure("confusion_matrix", fig)
            else:
                logger.info("Confusion matrix requested but no evaluation predictions found.")

        # ----- Attention heatmap -----
        if opts.get("attention_heatmap", False):
            from framework.analysis.plotting import plot_attention_heatmap
            # Requires the actual model – we don't have it here; skip gracefully
            logger.info("Attention heatmap requested but model not available in saver context. "
                        "To enable this, save the model first and call a separate analysis job.")

        # ----- Gradient flow -----
        if opts.get("gradient_flow", False):
            from framework.analysis.plotting import plot_gradient_flow
            # Requires gradient norms logged during training – can be extracted from history
            if any("grad_norm" in entry for entry in history):
                fig = plot_gradient_flow(history, figsize=figsize)
                _save_figure("gradient_flow", fig)

        # ----- Learning rate schedule -----
        if opts.get("lr_schedule", True):
            from framework.analysis.plotting import plot_lr_schedule
            if any("learning_rate" in entry for entry in history):
                fig = plot_lr_schedule(history, figsize=figsize)
                _save_figure("lr_schedule", fig)

        # ----- GPU memory / utilisation -----
        if opts.get("gpu_memory", False):
            from framework.analysis.plotting import plot_gpu_memory
            if any("gpu_mem" in entry for entry in history):
                fig = plot_gpu_memory(history, figsize=figsize)
                _save_figure("gpu_memory", fig)

        # ----- Training time vs steps -----
        if opts.get("training_time", True):
            from framework.analysis.plotting import plot_training_time
            fig = plot_training_time(history, figsize=figsize)
            _save_figure("training_time", fig)

        # ----- Parameter count (static) -----
        if opts.get("parameter_count", False):
            from framework.analysis.plotting import plot_parameter_distribution
            # This would need the model object; we skip with a warning
            logger.info("Parameter count plot requested but model not available.")

        # ----- Rank allocation (SVD) -----
        if opts.get("rank_allocation", False):
            from framework.analysis.plotting import plot_rank_allocation
            if any("rank" in entry for entry in history):
                fig = plot_rank_allocation(history, figsize=figsize)
                _save_figure("rank_allocation", fig)

        # ----- Pareto front -----
        if opts.get("pareto_front", False):
            from framework.analysis.plotting import plot_pareto_front
            # Needs multiple runs data; not available in single run context
            logger.info("Pareto front requires multiple runs comparison – skipped.")

        # ----- Radar chart -----
        if opts.get("radar_chart", False):
            from framework.analysis.plotting import plot_radar_chart
            # Requires multiple metrics; use final_metrics
            metrics_path = self.run_dir / "artifacts" / "final_metrics.json"
            if metrics_path.exists():
                import json
                with open(metrics_path) as f:
                    metrics = json.load(f)
                fig = plot_radar_chart(metrics, figsize=figsize)
                _save_figure("radar_chart", fig)