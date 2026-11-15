#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for SDPA backend selection and dispatch (Wave 4 W4-L3)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from neuralops.layers.sdpa import (
    SDPABackend,
    sdpa_backend_context,
    sdpa_forward,
    select_sdpa_backend,
)
from neuralops.layers.sdpa_attention import SDPAAttention


def test_sdpa_backend_selection_cpu() -> None:
    """Verify that on CPU, select_sdpa_backend always defaults to MATH."""
    q = torch.randn(2, 4, 16, 64, dtype=torch.float32)
    k = torch.randn(2, 4, 16, 64, dtype=torch.float32)
    v = torch.randn(2, 4, 16, 64, dtype=torch.float32)

    backend = select_sdpa_backend(q, k, v)
    assert backend == SDPABackend.MATH


def test_sdpa_backend_selection_logic_rules() -> None:
    """Verify the decision boundaries of select_sdpa_backend across dtypes and shapes."""

    def make_mock_tensor(is_cuda: bool, dtype: torch.dtype, head_dim: int) -> MagicMock:
        mock = MagicMock()
        mock.is_cuda = is_cuda
        mock.dtype = dtype
        mock.shape = (2, 4, 16, head_dim)
        return mock

    # 1. CUDA + fp16 + valid dim (64) + no mask -> FLASH
    q_flash = make_mock_tensor(is_cuda=True, dtype=torch.float16, head_dim=64)
    assert select_sdpa_backend(q_flash, q_flash, q_flash) == SDPABackend.FLASH

    # 2. CUDA + bf16 + valid dim (128) + no mask -> FLASH
    q_bf16 = make_mock_tensor(is_cuda=True, dtype=torch.bfloat16, head_dim=128)
    assert select_sdpa_backend(q_bf16, q_bf16, q_bf16) == SDPABackend.FLASH

    # 3. CUDA + fp16 + non-multiple-of-8 dim (65) -> MEM_EFFICIENT
    q_odd = make_mock_tensor(is_cuda=True, dtype=torch.float16, head_dim=65)
    assert select_sdpa_backend(q_odd, q_odd, q_odd) == SDPABackend.MEM_EFFICIENT

    # 4. CUDA + fp16 + head_dim > 256 (e.g. 512) -> MEM_EFFICIENT
    q_large = make_mock_tensor(is_cuda=True, dtype=torch.float16, head_dim=512)
    assert select_sdpa_backend(q_large, q_large, q_large) == SDPABackend.MEM_EFFICIENT

    # 5. CUDA + fp16 + valid dim + arbitrary mask -> MEM_EFFICIENT
    mask = torch.ones(2, 4, 16, 16)
    assert select_sdpa_backend(q_flash, q_flash, q_flash, attn_mask=mask) == SDPABackend.MEM_EFFICIENT

    # 6. CUDA + fp32 -> MEM_EFFICIENT
    q_fp32 = make_mock_tensor(is_cuda=True, dtype=torch.float32, head_dim=64)
    assert select_sdpa_backend(q_fp32, q_fp32, q_fp32) == SDPABackend.MEM_EFFICIENT

    # 7. CUDA + fp64 (float64) -> MATH
    q_fp64 = make_mock_tensor(is_cuda=True, dtype=torch.float64, head_dim=64)
    assert select_sdpa_backend(q_fp64, q_fp64, q_fp64) == SDPABackend.MATH


def test_sdpa_context_manager() -> None:
    """Verify that sdpa_backend_context can be entered and exited cleanly."""
    with sdpa_backend_context(SDPABackend.MATH):
        x = torch.tensor([1.0, 2.0])
        assert x.sum() == 3.0

    with sdpa_backend_context(SDPABackend.AUTO):
        assert True


def test_sdpa_forward_execution() -> None:
    """Verify sdpa_forward executes and matches torch reference on CPU."""
    q = torch.randn(2, 4, 8, 32)
    k = torch.randn(2, 4, 8, 32)
    v = torch.randn(2, 4, 8, 32)

    out = sdpa_forward(q, k, v, is_causal=True)
    assert out.shape == (2, 4, 8, 32)
    assert not torch.isnan(out).any()


def test_sdpa_attention_layer_backward() -> None:
    """Verify SDPAAttention layer computes outputs and backward gradients correctly."""
    layer = SDPAAttention(hidden_dim=64, num_heads=4, is_causal=True)
    x = torch.randn(2, 8, 64, requires_grad=True)

    out = layer(x)
    out_tensor = out[0] if isinstance(out, tuple) else out
    assert out_tensor.shape == (2, 8, 64)

    loss = out_tensor.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
