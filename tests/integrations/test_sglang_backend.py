#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for SGLang Linear Attention Backend and Registration.
"""

import pytest
import torch

from neuralops.integrations.sglang_backend import (
    NeuralOpsLinearAttnBackend,
    register,
)


def test_neuralops_linear_attn_backend_instantiation():
    backend = NeuralOpsLinearAttnBackend(
        num_heads=4,
        head_dim=32,
        state_dim=32,
        mechanism="gla_attention",
    )
    assert backend.num_heads == 4
    assert backend.head_dim == 32
    assert backend.state_dim == 32
    assert backend.uses_mamba_radix_cache is True
    assert backend.supports_mis is True


def test_neuralops_linear_attn_backend_forward_3d():
    backend = NeuralOpsLinearAttnBackend(num_heads=4, head_dim=32)
    batch_size, seq_len, hidden_dim = 2, 8, 4 * 32
    mixed_qkv = torch.randn(batch_size, seq_len, 3 * hidden_dim)

    out = backend.forward(mixed_qkv)
    assert out.shape == (batch_size, seq_len, hidden_dim)
    assert torch.isfinite(out).all()


def test_neuralops_linear_attn_backend_forward_tuple_and_gates():
    backend = NeuralOpsLinearAttnBackend(num_heads=4, head_dim=32)
    batch_size, seq_len, hidden_dim = 2, 8, 4 * 32
    q = torch.randn(batch_size, seq_len, hidden_dim)
    k = torch.randn(batch_size, seq_len, hidden_dim)
    v = torch.randn(batch_size, seq_len, hidden_dim)

    a = torch.sigmoid(torch.randn(batch_size, 4))
    b = torch.sigmoid(torch.randn(batch_size, 4))

    out = backend.forward((q, k, v), a=a, b=b)
    assert out.shape == (batch_size, seq_len, hidden_dim)
    assert torch.isfinite(out).all()

    # Test decode and extend aliases
    out_decode = backend.forward_decode((q[:, :1, :], k[:, :1, :], v[:, :1, :]), a=a, b=b)
    assert out_decode.shape == (batch_size, 1, hidden_dim)
    assert torch.isfinite(out_decode).all()

    out_extend = backend.forward_extend((q, k, v), a=a, b=b)
    assert out_extend.shape == (batch_size, seq_len, hidden_dim)
    assert torch.isfinite(out_extend).all()


def test_sglang_registration():
    res = register()
    # If sglang is installed, registration must succeed with uses_mamba_radix_cache=True
    try:
        from sglang.srt.configs.linear_attn_model_registry import _LINEAR_ATTN_MODEL_REGISTRY
        assert res is True
        spec = _LINEAR_ATTN_MODEL_REGISTRY[-1]
        assert spec.uses_mamba_radix_cache is True
        assert "NeuralOpsLinearAttnBackend" in spec.backend_class_name
    except ImportError:
        # If sglang is not present in local environment, it must return False without raising
        assert res is False
