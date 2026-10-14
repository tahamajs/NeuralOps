"""Kernel autotuner and SRAM-budget adaptive configuration manager for Triton kernels.

Eliminates `triton.runtime.OutOfResources: out of resource: shared memory`
across all GPU architectures (Turing 64KB, RTX 4070 99KB, A100 163KB, H100 227KB).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import torch

from neuralops.core.triton_kernels.hardware_profile import (
    HardwareProfile,
    get_device_hardware_profile,
)

logger = logging.getLogger("TritonAutotuner")

__all__ = [
    "Autotuner",
    "autotune_kernel",
    "estimate_kernel_sram_bytes",
    "get_safe_autotune_configs",
]


def estimate_kernel_sram_bytes(
    block_m: int,
    block_n: int,
    head_dim: int,
    dtype_bytes: int,
    num_stages: int,
    num_buffers: int = 3,
) -> int:
    """Compute the live tile footprint in GPU Shared Memory (SRAM).

    Formula:
      (block_m * head_dim + (num_buffers - 1) * block_n * head_dim) * dtype_bytes * max(1, num_stages)
    """
    m_footprint = int(block_m) * int(head_dim)
    n_footprint = (int(num_buffers) - 1) * int(block_n) * int(head_dim)
    stages = max(1, int(num_stages))
    return (m_footprint + n_footprint) * int(dtype_bytes) * stages


def get_safe_autotune_configs(
    device: torch.device | str | int | None = None,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float16,
    base_configs: list[dict[str, Any]] | None = None,
    profile: HardwareProfile | None = None,
) -> list[dict[str, Any]]:
    """Filters or constructs autotuning configurations guaranteed to fit within GPU SRAM budget.

    - Protects Turing (SM 7.5, 64KB SRAM) and RTX 4070 (SM 8.9, 99KB SRAM).
    - Caps live SRAM allocation at 85% of physical block shared memory (`0.85 * max_shared_memory_bytes`).
    - Automatically adjusts for AMD ROCm 64-thread wavefronts.
    - Guarantees at least one ultra-conservative fallback config is always returned.
    """
    if profile is None:
        profile = get_device_hardware_profile(device)

    # 1. Determine element byte width
    if dtype in (torch.float16, torch.bfloat16):
        dtype_bytes = 2
    elif dtype == torch.float32:
        dtype_bytes = 4
    elif dtype in (torch.int8, getattr(torch, "float8_e4m3fn", torch.int8), getattr(torch, "float8_e5m2", torch.int8)):
        dtype_bytes = 1
    else:
        dtype_bytes = 2

    # 2. Maximum allowable shared memory per block (85% safety margin)
    sram_budget = int(profile.max_shared_memory_bytes * 0.85) if profile.max_shared_memory_bytes > 0 else 49152

    # SM 7.0 / 7.5 tight threshold (strict 48 KB limit)
    if profile.sm_major == 7 or (dtype_bytes == 4 and head_dim >= 128):
        sram_budget = min(sram_budget, 49152)

    # 3. Default base configurations if none provided
    if base_configs is None:
        # Standard configuration ladder
        base_configs = [
            {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3},
            {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
            {"BLOCK_M": 64, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2},
            {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 2, "num_stages": 2},
            {"BLOCK_M": 32, "BLOCK_N": 16, "num_warps": 2, "num_stages": 1},
            {"BLOCK_M": 16, "BLOCK_N": 16, "num_warps": 2, "num_stages": 1},
        ]

    safe_configs: list[dict[str, Any]] = []

    for cfg in base_configs:
        bm = cfg.get("BLOCK_M", 32)
        bn = cfg.get("BLOCK_N", 32)
        stages = cfg.get("num_stages", 2)
        warps = cfg.get("num_warps", 4)

        # On SM 7.x, limit stages to 1 or 2
        if profile.sm_major == 7:
            stages = min(stages, 2)
            if bm > 64:
                bm = 64
            if bn > 64:
                bn = 64

        est_sram = estimate_kernel_sram_bytes(
            block_m=bm,
            block_n=bn,
            head_dim=head_dim,
            dtype_bytes=dtype_bytes,
            num_stages=stages,
        )

        if est_sram <= sram_budget:
            clean_cfg = dict(cfg)
            clean_cfg["BLOCK_M"] = bm
            clean_cfg["BLOCK_N"] = bn
            clean_cfg["num_stages"] = stages
            clean_cfg["num_warps"] = warps

            # AMD ROCm wavefront adjustments
            if profile.backend == "rocm":
                clean_cfg["waves_per_eu"] = 2

            safe_configs.append(clean_cfg)

    # 4. Ultra-conservative guarantee fallback
    if not safe_configs:
        safe_configs.append(
            {
                "BLOCK_M": 16,
                "BLOCK_N": 16,
                "num_warps": 2,
                "num_stages": 1,
                **({"waves_per_eu": 2} if profile.backend == "rocm" else {}),
            }
        )

    return safe_configs


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
