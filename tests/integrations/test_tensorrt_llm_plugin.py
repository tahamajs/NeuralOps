#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for TensorRT-LLM Plugin Module for NeuralOps Attention.
"""

import pytest
import torch

from neuralops.integrations.tensorrt_llm_plugin import (
    neuralops_attention,
    python_plugin,
)


def test_neuralops_attention_plugin_metadata():
    assert hasattr(neuralops_attention, "_trt_plugin_name")
    assert neuralops_attention._trt_plugin_name == "NeuralOpsAttention"
    assert hasattr(neuralops_attention, "_trt_outputs")
    assert neuralops_attention._trt_outputs == ["attention_output"]


def test_neuralops_attention_forward_4d():
    b, h, t, d = 2, 4, 16, 32
    q = torch.randn(b, h, t, d)
    k = torch.randn(b, h, t, d)
    v = torch.randn(b, h, t, d)

    out = neuralops_attention(q, k, v)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_neuralops_attention_forward_with_mask():
    b, h, t, d = 2, 4, 8, 32
    q = torch.randn(b, h, t, d)
    k = torch.randn(b, h, t, d)
    v = torch.randn(b, h, t, d)
    mask = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)

    out = neuralops_attention(q, k, v, mask=mask)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_neuralops_attention_forward_3d():
    b, t, d = 2, 8, 64
    q = torch.randn(b, t, d)
    k = torch.randn(b, t, d)
    v = torch.randn(b, t, d)

    out = neuralops_attention(q, k, v)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
