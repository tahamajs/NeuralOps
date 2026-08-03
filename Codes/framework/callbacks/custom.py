"""Custom training callbacks: EMA, SWA, gradient clipping, freeze/unfreeze, NaN check."""

import copy
import logging

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class EMACallback(TrainerCallback):
    """Exponential Moving Average of model parameters."""

    def __init__(self, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {}

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        """On train begin."""
        model = kwargs.get("model")
        if model is not None:
            self.shadow = {
                name: param.data.clone()
                for name, param in model.named_parameters()
                if param.requires_grad
            }

    def on_step_end(self, args, state, control, **kwargs) -> None:
        """On step end."""
        model = kwargs.get("model")
        if model is None:
            return
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def on_train_end(self, args, state, control, **kwargs) -> None:
        """On train end."""
        model = kwargs.get("model")
        if model is not None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(self.shadow[name])


class SWACallback(TrainerCallback):
    """Stochastic Weight Averaging (SWA)."""

    def __init__(self, start_epoch: int = 5, swa_lr: float = 1e-4) -> None:
        self.start_epoch = start_epoch
        self.swa_lr = swa_lr
        self.swa_model = None
        self.n_averaged = 0

    def on_epoch_end(self, args, state, control, **kwargs) -> None:
        """On epoch end."""
        model = kwargs.get("model")
        if model is None:
            return
        if state.epoch >= self.start_epoch:
            if self.swa_model is None:
                self.swa_model = copy.deepcopy(model).to(model.device)
                self.n_averaged = 1
            else:
                self.n_averaged += 1
                for swa_p, p in zip(self.swa_model.parameters(), model.parameters()):
                    swa_p.data = (
                        swa_p.data * (self.n_averaged - 1) + p.data
                    ) / self.n_averaged

    def on_train_end(self, args, state, control, **kwargs) -> None:
        """On train end."""
        model = kwargs.get("model")
        if self.swa_model is not None and model is not None:
            for p, swa_p in zip(model.parameters(), self.swa_model.parameters()):
                p.data.copy_(swa_p.data)
            logger.info("SWA: loaded averaged weights into model.")


class GradientClippingCallback(TrainerCallback):
    """Clips gradient norm after each step."""

    def __init__(self, max_norm: float = 1.0) -> None:
        self.max_norm = max_norm

    def on_step_end(self, args, state, control, **kwargs) -> None:
        """On step end."""
        model = kwargs.get("model")
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)


class FreezeLayersCallback(TrainerCallback):
    """Freezes specified layers until a given epoch."""

    def __init__(self, freeze_until_epoch: int = 3, layers_to_freeze: list = None) -> None:
        self.freeze_until_epoch = freeze_until_epoch
        self.layers_to_freeze = layers_to_freeze or []

    def on_epoch_begin(self, args, state, control, **kwargs) -> None:
        """On epoch begin."""
        model = kwargs.get("model")
        if model is None:
            return
        if state.epoch < self.freeze_until_epoch:
            for name, param in model.named_parameters():
                if any(layer in name for layer in self.layers_to_freeze):
                    param.requires_grad = False


class UnfreezeScheduleCallback(TrainerCallback):
    """Unfreezes all parameters at specified epochs."""

    def __init__(self, unfreeze_epochs: list = None) -> None:
        self.unfreeze_epochs = unfreeze_epochs or [2, 4, 6]

    def on_epoch_begin(self, args, state, control, **kwargs) -> None:
        """On epoch begin."""
        model = kwargs.get("model")
        if model is None:
            return
        if state.epoch in self.unfreeze_epochs:
            for param in model.parameters():
                param.requires_grad = True
            logger.info(f"Unfroze all layers at epoch {state.epoch}")


class NanCheckerCallback(TrainerCallback):
    """Stops training if loss becomes NaN or Inf."""

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        """On log."""
        import math

        if logs and "loss" in logs:
            if math.isnan(logs["loss"]) or math.isinf(logs["loss"]):
                logger.error("NaN or Inf loss detected, stopping training")
                control.should_training_stop = True


class ActivationCheckpointingCallback(TrainerCallback):
    """Enables gradient checkpointing on specified modules."""

    def __init__(self, modules_to_checkpoint: list = None) -> None:
        self.modules_to_checkpoint = modules_to_checkpoint or ["attention"]

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        """On train begin."""
        model = kwargs.get("model")
        if model is None:
            return
        for name, module in model.named_modules():
            if any(m in name for m in self.modules_to_checkpoint):
                if hasattr(module, "gradient_checkpointing_enable"):
                    module.gradient_checkpointing_enable()
