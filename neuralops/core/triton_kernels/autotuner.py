"""Kernel autotuner for Triton kernels and attention operations."""
from __future__ import annotations
from typing import Callable, Any
import time
import torch

__all__ = ["Autotuner", "autotune_kernel"]


class Autotuner:
    """Benchmarks multiple kernel configurations and picks the best performing parameters."""

    def __init__(self, warmup: int = 2, rep: int = 5) -> None:
        self.warmup = warmup
        self.rep = rep

    def tune(
        self,
        configs: list[dict[str, Any]],
        fn: Callable[[dict[str, Any]], Any],
    ) -> tuple[dict[str, Any], float]:
        """Runs fn with each config in configs and returns (best_config, best_time_ms)."""
        best_cfg = configs[0]
        best_time = float("inf")

        for cfg in configs:
            try:
                # Warmup
                for _ in range(self.warmup):
                    fn(cfg)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Timing
                start = time.perf_counter()
                for _ in range(self.rep):
                    fn(cfg)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed_ms = (time.perf_counter() - start) / self.rep * 1000.0
                if elapsed_ms < best_time:
                    best_time = elapsed_ms
                    best_cfg = cfg
            except Exception:
                continue

        return best_cfg, best_time


def autotune_kernel(
    configs: list[dict[str, Any]],
    fn: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    """Helper to return the best configuration dictionary."""
    tuner = Autotuner()
    best_cfg, _ = tuner.tune(configs, fn)
    return best_cfg
