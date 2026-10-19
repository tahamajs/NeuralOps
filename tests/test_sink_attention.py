# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for sink attention."""
import torch
import pytest
from neuralops.attentions.adaptive.sink_attention import SinkAttention


def test_sink_attention_no_op():
    layer = SinkAttention(d_model=16, num_heads=2, num_sink_tokens=4, window_size=64)
    x = torch.randn(2, 32, 16)
    out, _ = layer(x)
    assert out.shape == (2, 32, 16)


def test_sink_attention_clamping():
    layer = SinkAttention(d_model=16, num_heads=2, num_sink_tokens=4, window_size=32)
    x = torch.randn(2, 64, 16)
    out, _ = layer(x)
    assert out.shape == (2, 64, 16)
    assert torch.isfinite(out).all()
