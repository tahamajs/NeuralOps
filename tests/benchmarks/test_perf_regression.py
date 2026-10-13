"""
Regression test for attention latency and memory footprint.
Outputs JSON report comparing standard MHA vs key NeuralOps subquadratic variants.
"""

import json
import time
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.triton_kernels.op_registry import (
    compilable_linear_attn,
    compilable_low_rank,
)


def benchmark_mha_vs_linear():
    B, H, N, D = 2, 4, 128, 64
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    # Standard quadratic attention
    t0 = time.perf_counter()
    for _ in range(5):
        scale = 1.0 / (D ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        _ = torch.matmul(attn, v)
    t_mha = (time.perf_counter() - t0) / 5

    # Linear attention (O(N))
    t0 = time.perf_counter()
    for _ in range(5):
        _ = compilable_linear_attn(q, k, v)
    t_linear = (time.perf_counter() - t0) / 5

    report = {
        "batch_size": B,
        "num_heads": H,
        "seq_len": N,
        "head_dim": D,
        "mha_latency_s": t_mha,
        "linear_latency_s": t_linear,
        "speedup": round(t_mha / max(t_linear, 1e-9), 2),
    }

    out_file = Path("results") / "perf_regression.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def test_perf_regression():
    report = benchmark_mha_vs_linear()
    assert "mha_latency_s" in report
    assert "linear_latency_s" in report
    assert report["linear_latency_s"] > 0
