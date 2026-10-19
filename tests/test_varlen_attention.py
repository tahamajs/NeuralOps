#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for variable-length attention."""
import torch
import pytest
from neuralops.core.triton_kernels.varlen_attention import varlen_attention_pytorch


def test_varlen_attention_shapes():
    cu_seqlens = torch.tensor([0, 10, 30], dtype=torch.int32)
    H, D = 4, 32
    total_tokens = 30

    q = torch.randn(total_tokens, H, D)
    k = torch.randn(total_tokens, H, D)
    v = torch.randn(total_tokens, H, D)

    out = varlen_attention_pytorch(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=20,
        max_seqlen_k=20,
        causal=True,
    )
    assert out.shape == (total_tokens, H, D)
