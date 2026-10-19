#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Analytical roofline metrics for attention forward passes."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def calculate_flops(attention_type: str, B: int, H: int, S: int, D: int, *, window: int | None = None) -> int:
    """Return score/value multiply-add FLOPs for an attention forward pass."""
    if min(B, H, S, D) <= 0:
        raise ValueError("B, H, S, and D must be positive")
    kind = attention_type.lower()
    if kind == "full":
        pairs = S * S
    elif kind == "causal":
        pairs = S * (S + 1) // 2
    elif kind == "local":
        if window is None or window <= 0:
            raise ValueError("local attention requires a positive window")
        pairs = sum(min(S, i + window + 1) for i in range(S))
    elif kind == "mla":
        pairs = S * S
        D = max(1, D // 2)
    else:
        raise ValueError(f"Unsupported attention type: {attention_type}")
    return int(4 * B * H * pairs * D)


def calculate_bandwidth_bytes(attention_type: str, B: int, H: int, S: int, D: int, dtype: str | Any = "float16", *, window: int | None = None) -> int:
    """Estimate bytes moved for Q, K, V, and output tensors."""
    del attention_type, window
    if min(B, H, S, D) <= 0:
        raise ValueError("B, H, S, and D must be positive")
    itemsize = getattr(dtype, "itemsize", None) or {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}.get(str(dtype), 2)
    return int(4 * B * H * S * D * itemsize)


def roofline_metrics(attention_type: str, B: int, H: int, S: int, D: int, latency_ms: float, *, window: int | None = None, dtype: str = "float16", dram_gbps: float = 256.0, peak_tflops: float = 79.2) -> dict[str, float | int | str]:
    """Compute arithmetic intensity and achieved throughput."""
    flops = calculate_flops(attention_type, B, H, S, D, window=window)
    bandwidth = calculate_bandwidth_bytes(attention_type, B, H, S, D, dtype, window=window)
    seconds = latency_ms / 1000.0
    return {
        "attention_type": attention_type,
        "flops": flops,
        "bytes": bandwidth,
        "arithmetic_intensity": flops / bandwidth,
        "achieved_tflops": flops / seconds / 1e12 if seconds > 0 else 0.0,
        "achieved_gbps": bandwidth / seconds / 1e9 if seconds > 0 else 0.0,
        "compute_roof_tflops": peak_tflops,
        "bandwidth_roof_gbps": dram_gbps,
    }


def plot_roofline(latencies: list[float], flops: list[float], bandwidth: list[float], output: str | Path) -> Path:
    """Write a log-log roofline scatter plot from measured samples."""
    import matplotlib.pyplot as plt

    intensities = [f / b for f, b in zip(flops, bandwidth, strict=True)]
    achieved = [f / (t / 1000.0) / 1e12 for f, t in zip(flops, latencies, strict=True)]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(intensities, achieved, "o")
    ax.set_xlabel("Arithmetic intensity (FLOPs/byte)")
    ax.set_ylabel("Achieved throughput (TFLOP/s)")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run_benchmark(device: str = "cuda") -> dict[str, Any]:
    """Return a deterministic analytical causal-attention sample."""
    del device
    return roofline_metrics("causal", 4, 12, 512, 64, 1.0)
