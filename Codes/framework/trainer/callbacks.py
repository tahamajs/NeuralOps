import os
import time

import torch
from transformers import TrainerCallback


class MemoryCallback(TrainerCallback):
    """Logs GPU/MPS memory usage."""

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        """On log."""
        if logs is None:
            return
        if torch.cuda.is_available():
            logs["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024 / 1024
            logs["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024 / 1024
        if torch.backends.mps.is_available():
            logs["mps_memory"] = torch.mps.current_allocated_memory() / 1024 / 1024


class TimeCallback(TrainerCallback):
    """Logs elapsed time."""

    def __init__(self) -> None:
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        """On train begin."""
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        """On log."""
        if logs is not None and self.start_time is not None:
            logs["elapsed_time"] = time.time() - self.start_time


class GradientNormCallback(TrainerCallback):
    """Logs gradient norms."""

    def __init__(self, log_every_n_steps: int = 100) -> None:
        self.log_every_n_steps = log_every_n_steps

    def on_step_end(self, args, state, control, model=None, **kwargs) -> None:
        """On step end."""
        if model is None or state.global_step % self.log_every_n_steps != 0:
            return
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        # logs will be passed to on_log? Actually we can just log to the trainer
        if hasattr(args, "logging_dir"):
            pass  # We can't easily log here; the SVDAwareTrainer logs it automatically
        # For simplicity we just print
        print(f"Step {state.global_step}: grad norm = {total_norm:.4f}")


class SVDAttentionCallback(TrainerCallback):
    """Logs SVD compression statistics."""

    def __init__(self, log_attention_stats: bool = True) -> None:
        self.log_attention_stats = log_attention_stats

    def on_log(self, args, state, control, logs=None, model=None, **kwargs) -> None:
        """On log."""
        if not self.log_attention_stats or logs is None or model is None:
            return
        svd_layers = 0
        compressed_params = 0
        total_params = 0
        for module in model.modules():
            total_params += sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            name = module.__class__.__name__
            if name in [
                "SVDLinearLayer",
                "CompressedAttentionLayer",
                "FactorisedLinear",
            ]:
                svd_layers += 1
                compressed_params += sum(p.numel() for p in module.parameters())
        if svd_layers > 0:
            logs["svd_layers"] = svd_layers
            logs["compression_ratio"] = (
                compressed_params / total_params if total_params > 0 else 0
            )


class PreemptionCallback(TrainerCallback):
    """Stops training when a preemption flag file is present."""

    def __init__(self, preempt_flag_dir="/tmp/preempt_flags", job_id=None) -> None:
        self.preempt_flag_dir = preempt_flag_dir
        self.job_id = job_id

    def on_step_end(self, args, state, control, **kwargs) -> None:
        """On step end."""
        if self.job_id and os.path.exists(f"{self.preempt_flag_dir}/{self.job_id}"):
            print("Preemption requested, stopping training.")
            control.should_training_stop = True
