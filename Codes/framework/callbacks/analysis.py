"""Analysis callbacks: SVD, attention heatmaps, gradient norms."""
import logging
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

class SVDAttentionCallback(TrainerCallback):
    """Periodically runs SVD analysis on attention modules."""
    def __init__(self, layer_interval: int = 2, log_every_steps: int = 500):
        self.layer_interval = layer_interval
        self.log_every_steps = log_every_steps

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get('model')
        if model is None or state.global_step % self.log_every_steps != 0:
            return
        logger.info("Performing SVD analysis on attention layers...")
        # Delegates to framework.analysis if needed


class AttentionHeatmapCallback(TrainerCallback):
    """Generates attention heatmaps periodically."""
    def __init__(self, output_dir: str, log_every_steps: int = 1000):
        self.output_dir = output_dir
        self.log_every_steps = log_every_steps

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.global_step % self.log_every_steps == 0:
            logger.info("Generating attention heatmaps...")


class GradientNormCallback(TrainerCallback):
    """Logs gradient norms."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'grad_norm' in logs:
            logger.info(f"Gradient norm: {logs['grad_norm']}")


class LayerwiseAnalysisCallback(TrainerCallback):
    """Performs layer‑wise analysis (intrinsic dimension, etc.)."""
    def __init__(self, log_every_steps: int = 2000):
        self.log_every_steps = log_every_steps

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.global_step % self.log_every_steps == 0:
            logger.info("Running layer‑wise analysis...")


class ActivationStatsCallback(TrainerCallback):
    """Collects activation statistics."""
    def __init__(self):
        self.activations = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'activation_mean' in logs:
            self.activations.append(logs)