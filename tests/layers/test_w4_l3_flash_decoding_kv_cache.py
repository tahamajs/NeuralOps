#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for FlashDecoding past_key_values cache threading."""

import pytest
import torch

from neuralops.layers.flash_decoding import FlashDecodingAttention, flash_decoding_forward


def test_flash_decoding_forward_reference():
    """Verify flash_decoding_forward matches PyTorch math reference."""
    b, h, d = 2, 4, 32
    n_q = 1
    n_kv = 256
    q = torch.randn(b, h, n_q, d)
    k = torch.randn(b, h, n_kv, d)
    v = torch.randn(b, h, n_kv, d)

    out = flash_decoding_forward(q, k, v, chunk_size=64)
    assert out.shape == (b, h, n_q, d)
    assert torch.isfinite(out).all()


def test_flash_decoding_past_key_values_threading():
    """Verify past_key_values is threaded across sequential decoding steps."""
    b, dim = 2, 64
    layer = FlashDecodingAttention(dim=dim, num_heads=4, chunk_size=32)

    # Step 1: Initial context token (prefill, length 8)
    x_init = torch.randn(b, 8, dim)
    out_init, past_kv = layer(x_init, use_cache=True)
    assert out_init.shape == (b, 8, dim)
    assert past_kv is not None
    assert past_kv[0].shape[-2] == 8  # Key cache length == 8
    assert past_kv[1].shape[-2] == 8  # Value cache length == 8

    # Step 2: Autoregressive decoding step 1 (N_q = 1)
    x_step1 = torch.randn(b, 1, dim)
    out_step1, past_kv = layer(x_step1, past_key_value=past_kv, use_cache=True)
    assert out_step1.shape == (b, 1, dim)
    assert past_kv[0].shape[-2] == 9  # Key cache length == 9
    assert past_kv[1].shape[-2] == 9

    # Step 3: Autoregressive decoding step 2 (N_q = 1)
    x_step2 = torch.randn(b, 1, dim)
    out_step2, past_kv = layer(x_step2, past_key_value=past_kv, use_cache=True)
    assert out_step2.shape == (b, 1, dim)
    assert past_kv[0].shape[-2] == 10  # Key cache length == 10
    assert past_kv[1].shape[-2] == 10
    assert torch.isfinite(out_step2).all()
