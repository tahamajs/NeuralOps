"""Compare NeuralOps modules with FLA implementations.

This requires FLA to be installed:
    pip install flash-linear-attention[cuda]
"""
from __future__ import annotations

import sys
import time
from typing import Any

import torch

def benchmark_neuralops(module_name: str, B: int = 2, N: int = 512, D: int = 64, H: int = 4):
    """Benchmark a NeuralOps attention module."""
    from neuralops.core.registry import ATTENTION_REGISTRY
    import neuralops.attentions  # noqa: F401

    cls = ATTENTION_REGISTRY.get(module_name)
    if cls is None:
        return None

    try:
        attn = cls(hidden_size=H * D, num_heads=H)
    except Exception:
        return None

    x = torch.randn(B, N, H * D)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            try:
                attn(x)
            except Exception:
                return None

    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        with torch.no_grad():
            try:
                attn(x)
            except Exception:
                return None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 20
    return elapsed * 1000


def benchmark_fla(module_name: str, B: int = 2, N: int = 512, D: int = 64, H: int = 4):
    """Benchmark an FLA module (if FLA is installed)."""
    try:
        import fla  # noqa: F401
    except ImportError:
        return None

    try:
        if module_name == "gla":
            from fla.layers import GatedLinearAttention
            layer = GatedLinearAttention(hidden_size=H * D, num_heads=H)
        elif module_name == "retnet":
            from fla.layers import MultiScaleRetention
            layer = MultiScaleRetention(hidden_size=H * D, num_heads=H)
        else:
            return None
    except ImportError:
        return None

    x = torch.randn(B, N, H * D)
    for _ in range(3):
        with torch.no_grad():
            try:
                layer(x)
            except Exception:
                return None

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        with torch.no_grad():
            try:
                layer(x)
            except Exception:
                return None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return ((time.perf_counter() - start) / 20) * 1000


if __name__ == "__main__":
    comparisons = [
        ("gated_linear_attention", "gla"),
        ("decay_linear_attention", "retnet"),
    ]

    print(f"{'Module':<30} {'NeuralOps (ms)':<15} {'FLA (ms)':<15} {'Ratio':<10}")
    print("=" * 75)
    for no_name, fla_name in comparisons:
        no_time = benchmark_neuralops(no_name)
        fla_time = benchmark_fla(fla_name)
        if no_time is None:
            print(f"{no_name:<30} {'N/A':<15} {str(fla_time):<15} {'-':<10}")
            continue
        if fla_time is None:
            print(f"{no_name:<30} {no_time:<15.2f} {'N/A':<15} {'-':<10}")
            continue
        ratio = no_time / fla_time
        print(f"{no_name:<30} {no_time:<15.2f} {fla_time:<15.2f} {ratio:<10.2f}")
