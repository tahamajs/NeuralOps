import logging
from pathlib import Path
from transformers import TrainerCallback
from .tracker import MetricsTracker

class LoggingCallback(TrainerCallback):
    """
    HuggingFace Trainer callback that pushes metrics to a MetricsTracker.
    Optionally generates plots after training ends (delegates to analysis).
    """
    def __init__(
        self,
        tracker: MetricsTracker,
        logger: logging.Logger,
        plots_dir: str = None,
        save_plots: bool = False,
    ):
        self.tracker = tracker
        self.logger = logger
        self.plots_dir = Path(plots_dir) if plots_dir else None
        self.save_plots = save_plots
        self._train_losses = []
        self._eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            self.tracker.log(logs, step=step)
            # Human‑readable summary
            loss = logs.get("loss", None)
            eval_loss = logs.get("eval_loss", None)
            lr = logs.get("learning_rate", None)
            msg = f"Step {step}: "
            if loss is not None: msg += f"loss={loss:.4f} "
            if eval_loss is not None: msg += f"eval_loss={eval_loss:.4f} "
            if lr is not None: msg += f"lr={lr:.2e}"
            self.logger.info(msg)

    def on_train_end(self, args, state, control, **kwargs):
        self.tracker.flush()
        self.logger.info("Training completed. Metrics flushed.")
        if self.save_plots and self.plots_dir:
            try:
                from framework.analysis.plotting import plot_training_history
                df = self.tracker.to_dataframe()
                self.plots_dir.mkdir(parents=True, exist_ok=True)
                plot_training_history(
                    {"train_steps": df["step"].tolist() if "step" in df.columns else [],
                     "train_loss": df["loss"].tolist() if "loss" in df.columns else []},
                    save_path=str(self.plots_dir / "training_history.png")
                )
                self.logger.info(f"Plots saved to {self.plots_dir}")
            except Exception as e:
                self.logger.warning(f"Plot generation failed: {e}")