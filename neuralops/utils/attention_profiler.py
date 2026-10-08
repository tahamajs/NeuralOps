"""Attention Memory Profiler to measure peak activation memory and time overhead."""
from __future__ import annotations
from typing import Callable, Any
import time
import torch
import torch.nn as nn

__all__ = ["AttentionProfiler", "profile_attention_layer"]


class AttentionProfiler:
    """Profiles memory and execution latency of attention operations."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)

    def profile(
        self,
        module: Callable[..., Any],
        *args: Any,
        num_warmup: int = 2,
        num_iters: int = 5,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Profiles forward pass of an attention module.

        Returns:
            Dict containing mean latency in ms and peak memory in MB (if CUDA).
        """
        is_cuda = self.device.type == "cuda" and torch.cuda.is_available()

        # Warmup
        for _ in range(num_warmup):
            _ = module(*args, **kwargs)

        if is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        start_time = time.perf_counter()
        for _ in range(num_iters):
            _ = module(*args, **kwargs)

        if is_cuda:
            torch.cuda.synchronize(self.device)

        elapsed = (time.perf_counter() - start_time) / num_iters * 1000.0  # ms

        peak_mem_mb = 0.0
        if is_cuda:
            peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        return {
            "latency_ms": elapsed,
            "peak_memory_mb": peak_mem_mb,
            "num_iters": float(num_iters),
        }


def profile_attention_layer(layer: nn.Module, sample_input: torch.Tensor) -> dict[str, float]:
    """Helper function to profile a given layer with sample input."""
    profiler = AttentionProfiler(sample_input.device)
    return profiler.profile(layer, sample_input)
