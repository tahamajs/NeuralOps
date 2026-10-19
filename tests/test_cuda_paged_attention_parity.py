#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Parity, edge case, and fallback tests for PagedAttention (Module 1).
"""

import pytest
import torch

from neuralops.core.cuda_kernels.paged_attention import (
    paged_attention,
    paged_attention_pytorch_reference,
)


def _setup_paged_inputs(
    num_seqs=4,
    num_heads=8,
    num_kv_heads=2,
    head_dim=64,
    block_size=16,
    max_blocks_per_seq=16,
    device="cuda",
    dtype=torch.float32,
):
    torch.manual_seed(42)
    q = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=dtype)
    num_total_blocks = num_seqs * max_blocks_per_seq
    k_cache = torch.randn(num_total_blocks, num_kv_heads, head_dim, block_size, device=device, dtype=dtype)
    v_cache = torch.randn(num_total_blocks, num_kv_heads, head_dim, block_size, device=device, dtype=dtype)

    context_lens = torch.tensor([15, 32, 64, 128], device=device, dtype=torch.int32)
    block_tables = torch.zeros(num_seqs, max_blocks_per_seq, device=device, dtype=torch.int32)

    for i in range(num_seqs):
        n_blocks = (int(context_lens[i].item()) + block_size - 1) // block_size
        block_tables[i, :n_blocks] = torch.arange(i * max_blocks_per_seq, i * max_blocks_per_seq + n_blocks, device=device)

    return q, k_cache, v_cache, block_tables, context_lens, block_size


def test_paged_attention_forward_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for PagedAttention test")

    q, k_cache, v_cache, block_tables, context_lens, block_size = _setup_paged_inputs()

    out_cuda = paged_attention(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        block_size=block_size,
    )

    out_ref = paged_attention_pytorch_reference(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        block_size=block_size,
        max_context_len=int(context_lens.max().item()),
        scale=1.0 / (q.shape[-1] ** 0.5),
    )

    torch.testing.assert_close(out_cuda, out_ref, atol=1e-3, rtol=1e-3)


def test_paged_attention_edge_empty_seq():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for PagedAttention test")

    q, k_cache, v_cache, block_tables, context_lens, block_size = _setup_paged_inputs()
    context_lens[0] = 0

    out_cuda = paged_attention(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        block_size=block_size,
    )
    assert not torch.isnan(out_cuda).any()
    assert torch.all(out_cuda[0] == 0.0)
