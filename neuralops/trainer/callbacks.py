#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Trainer Callbacks.

Provides training event callbacks for memory tracking, checkpoint management,
and progress logging with HuggingFace Trainer compatibility.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger("neuralops.trainer.callbacks")


class MemoryCallback(TrainerCallback):
    """Logs GPU and MPS memory usage during logging and evaluation events."""

    def _collect_memory_stats(self) -> Dict[str, float]:
        """Collect current device memory metrics in megabytes.

        Returns:
            Dictionary of memory metrics in megabytes.
        """
        stats: Dict[str, float] = {}
        if torch.cuda.is_available():
            try:
                stats["gpu_memory_allocated"] = torch.cuda.memory_allocated() / (1024 * 1024)
                stats["gpu_memory_reserved"] = torch.cuda.memory_reserved() / (1024 * 1024)
                stats["gpu_peak_memory"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            except Exception as e:
                logger.debug("Failed to query CUDA memory stats: %s", e)
        if torch.backends.mps.is_available():
            try:
                stats["mps_memory"] = torch.mps.current_allocated_memory() / (1024 * 1024)
            except Exception as e:
                logger.debug("Failed to query MPS memory stats: %s", e)
        return stats

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Inject memory metrics into training log dict.

        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control flow state.
            logs: Output log dictionary.
            **kwargs: Extra event arguments.
        """
        if logs is None:
            return
        logs.update(self._collect_memory_stats())

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Inject memory metrics into evaluation metrics dict.

        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control flow state.
            metrics: Output metrics dictionary.
            **kwargs: Extra event arguments.
        """
        if metrics is None:
            return
        metrics.update(self._collect_memory_stats())

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Handle train end event.

        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control flow state.
            **kwargs: Extra event arguments.
        """


class TimeCallback(TrainerCallback):
    """Logs elapsed wall-clock training and evaluation time."""

    def __init__(self) -> None:
        """Initialize TimeCallback."""
        self.start_time: Optional[float] = None
        self.total_time: Optional[float] = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Record training start time.

        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control flow state.
            **kwargs: Extra event arguments.
        """
        self.start_time = time.time()

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Inject elapsed training time into log dict.

        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control flow state.
            logs: Output log dictionary.
            **kwargs: Extra event arguments.
        """
        if logs is not None and self.start_time is not None:
            logs["elapsed_time"] = time.time() - self.start_time

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Inject elapsed time into evaluation metrics dict.

        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control flow state.
            metrics: Output metrics dictionary.
            **kwargs: Extra event arguments.
        """
        if metrics is not None and self.start_time is not None:
            metrics["elapsed_time"] = time.time() - self.start_time

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Record final total training time.

        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control flow state.
            **kwargs: Extra event arguments.
        """
        if self.start_time is not None:
            self.total_time = time.time() - self.start_time


class GradientNormCallback(TrainerCallback):
    """Tracks and logs global gradient norms across training steps."""

    def __init__(self, log_every_n_steps: int = 100) -> None:
        """Initialize GradientNormCallback.

        Args:
            log_every_n_steps: Interval between step gradient norm measurements.
        """
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.last_grad_norm: Optional[float] = None

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        """Calculate and store global gradient norm.

        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control flow state.
            model: Current model being trained.
            **kwargs: Extra event arguments.
        """
        target_model = model or kwargs.get("model")
        if target_model is None:
            return
        if state.global_step % self.log_every_n_steps != 0 and state.global_step != 0:
            return

        total_norm_sq = 0.0
        for p in target_model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().norm(2)
                total_norm_sq += float(param_norm.item() ** 2)

        self.last_grad_norm = total_norm_sq**0.5
        logger.debug("Step %d: grad norm = %.4f", state.global_step, self.last_grad_norm)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Inject last gradient norm into logs.

        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control flow state.
            logs: Output log dictionary.
            **kwargs: Extra event arguments.
        """
        if logs is not None and self.last_grad_norm is not None:
            logs["grad_norm"] = self.last_grad_norm

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Inject last gradient norm into evaluation metrics.

        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control flow state.
            metrics: Output metrics dictionary.
            **kwargs: Extra event arguments.
        """
        if metrics is not None and self.last_grad_norm is not None:
            metrics["grad_norm"] = self.last_grad_norm

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Handle train end event.

        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control flow state.
            **kwargs: Extra event arguments.
        """


class SVDAttentionCallback(TrainerCallback):
    """Logs SVD compression statistics and parameter ratios."""

    def __init__(self, log_attention_stats: bool = True) -> None:
        """Initialize SVDAttentionCallback.

        Args:
            log_attention_stats: Whether to compute and log SVD attention statistics.
        """
        self.log_attention_stats = log_attention_stats
        self.last_svd_layers: int = 0
        self.last_compression_ratio: float = 0.0

    def _compute_stats(self, model: nn.Module) -> Tuple[int, float]:
        """Compute SVD layer count and true parameter compression ratio.

        Args:
            model: PyTorch model module.

        Returns:
            Tuple of (svd_layers_count, compression_ratio).
        """
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if total_params == 0:
            total_params = sum(p.numel() for p in model.parameters())

        svd_layers = 0
        compressed_params = 0
        svd_class_names = {
            "SVDLinearLayer",
            "CompressedAttentionLayer",
            "FactorisedLinear",
            "DummySVDLinearLayer",
        }

        for module in model.modules():
            name = module.__class__.__name__
            if name in svd_class_names or "SVD" in name:
                svd_layers += 1
                compressed_params += sum(p.numel() for p in module.parameters())

        ratio = (compressed_params / total_params) if total_params > 0 else 0.0
        return svd_layers, ratio

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        model: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        """Log SVD compression statistics.

        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control flow state.
            logs: Output log dictionary.
            model: Current model.
            **kwargs: Extra event arguments.
        """
        if not self.log_attention_stats or logs is None:
            return
        target_model = model or kwargs.get("model")
        if target_model is None:
            return

        svd_layers, ratio = self._compute_stats(target_model)
        self.last_svd_layers = svd_layers
        self.last_compression_ratio = ratio

        if svd_layers > 0:
            logs["svd_layers"] = svd_layers
            logs["compression_ratio"] = ratio

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, Any]] = None,
        model: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        """Inject SVD compression statistics into evaluation metrics.

        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control flow state.
            metrics: Output metrics dictionary.
            model: Current model.
            **kwargs: Extra event arguments.
        """
        if not self.log_attention_stats or metrics is None:
            return
        target_model = model or kwargs.get("model")
        if target_model is None:
            return

        svd_layers, ratio = self._compute_stats(target_model)
        self.last_svd_layers = svd_layers
        self.last_compression_ratio = ratio

        if svd_layers > 0:
            metrics["svd_layers"] = svd_layers
            metrics["compression_ratio"] = ratio

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        """Handle train end event and record final SVD stats.

        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control flow state.
            model: Current model.
            **kwargs: Extra event arguments.
        """
        target_model = model or kwargs.get("model")
        if target_model is not None:
            svd_layers, ratio = self._compute_stats(target_model)
            self.last_svd_layers = svd_layers
            self.last_compression_ratio = ratio


class PreemptionCallback(TrainerCallback):
    """Safely halts training and triggers checkpoint saving upon preemption signal."""

    def __init__(
        self,
        preempt_flag_dir: str = "/tmp/preempt_flags",
        job_id: Optional[str] = None,
    ) -> None:
        """Initialize PreemptionCallback.

        Args:
            preempt_flag_dir: Directory to monitor for preemption flag files.
            job_id: Job identifier (defaults to SLURM_JOB_ID or JOB_ID env vars if None).
        """
        self.preempt_flag_dir = preempt_flag_dir
        self.job_id = job_id

    def _resolve_job_id(self) -> Optional[str]:
        """Resolve current job ID from parameter or environment.

        Returns:
            Resolved job ID string or None.
        """
        if self.job_id:
            return str(self.job_id)
        return os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID")

    def _is_preempted(self) -> bool:
        """Check if preemption has been requested.

        Returns:
            True if preemption flag file exists, False otherwise.
        """
        job_id = self._resolve_job_id()
        if job_id:
            flag_path = os.path.join(self.preempt_flag_dir, job_id)
            if os.path.exists(flag_path):
                return True
        generic_path = os.path.join(self.preempt_flag_dir, "preempt")
        return os.path.exists(generic_path)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Check preemption flag at step end and request safe stop + save.

        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control flow state.
            **kwargs: Extra event arguments.
        """
        if self._is_preempted():
            logger.warning("Preemption signal detected at step %d. Saving checkpoint and stopping.", state.global_step)
            control.should_training_stop = True
            control.should_save = True


__all__ = [
    "MemoryCallback",
    "TimeCallback",
    "GradientNormCallback",
    "SVDAttentionCallback",
    "PreemptionCallback",
]
