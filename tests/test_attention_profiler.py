# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for AttentionProfiler."""
import torch
import torch.nn as nn
from neuralops.utils.attention_profiler import AttentionProfiler, profile_attention_layer


def test_attention_profiler():
    lin = nn.Linear(16, 16)
    x = torch.randn(4, 16)
    profiler = AttentionProfiler()
    res = profiler.profile(lin, x, num_warmup=1, num_iters=2)

    assert "latency_ms" in res
    assert res["latency_ms"] >= 0.0
    assert "peak_memory_mb" in res
    assert res["num_iters"] == 2.0


def test_profile_attention_layer_helper():
    lin = nn.Linear(8, 8)
    x = torch.randn(2, 8)
    res = profile_attention_layer(lin, x)
    assert res["latency_ms"] >= 0.0
