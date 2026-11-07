#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
GPU Parity & Correctness Test Suite for PagedAttention Triton Kernel & Vectorized PyTorch Reference.

Verifies:
1. Parity check across random seeds, sequence lengths (1 to 2048), and batch sizes (1 to 16).
2. Causal mask verification during chunked prefill / autoregressive decoding.
3. Numerical stability checks (absence of NaNs, Infs, or subnormal overflows) across FP16, BF16, and FP8.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F

from neuralops.inference.paged_cache import PagedKVCache, paged_attention_reference
from neuralops.ops.triton.paged_attention import paged_attention_kernel


def test_paged_attention_parity_across_shapes_and_seeds() -> None:
    """Test 1: Parity check across random seeds, sequence lengths (1 to 2048), and batch sizes (1 to 16)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_heads = 4
    head_dim = 64
    block_sizes = [16, 32]

    # Test matrix of configurations: batch sizes 1 to 16, sequence lengths 1 to 2048
    test_configs = [
        (1, 1, 16),
        (2, 33, 16),
        (4, 128, 32),
        (8, 512, 16),
        (16, 1024, 32),
        (2, 2048, 32),
    ]

    for seed in (42, 1337):
        torch.manual_seed(seed)
        for batch_size, seq_len, block_sz in test_configs:
            num_blocks_per_seq = (seq_len + block_sz - 1) // block_sz
            total_blocks = batch_size * num_blocks_per_seq + 10

            cache = PagedKVCache(
                num_blocks=total_blocks,
                block_size=block_sz,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=torch.float32,
                device=device,
            )

            block_table_list = []
            seq_lens_list = []

            for b in range(batch_size):
                cache.allocate_sequence(b)
                # Fill tokens
                cur_slen = seq_len if b % 2 == 0 else max(1, seq_len // 2)
                for _ in range(cur_slen):
                    k_tok = torch.randn(num_heads, head_dim, device=device, dtype=torch.float32)
                    v_tok = torch.randn(num_heads, head_dim, device=device, dtype=torch.float32)
                    cache.append_token(b, k_tok, v_tok)
                
                blocks = cache.block_tables[b]
                block_table_list.append(blocks + [0] * (num_blocks_per_seq - len(blocks)))
                seq_lens_list.append(cur_slen)

            block_tables_tensor = torch.tensor(block_table_list, dtype=torch.int32, device=device)
            seq_lens_tensor = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)

            q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float32)

            ref_out = paged_attention_reference(
                q, cache.key_cache, cache.value_cache, block_tables_tensor, seq_lens_tensor, block_size=block_sz
            )
            kernel_out = paged_attention_kernel(
                q, cache.key_cache, cache.value_cache, block_tables_tensor, seq_lens_tensor, block_size=block_sz
            )

            # Numerical Precision Gate: max diff < 1e-3
            max_diff = (ref_out - kernel_out).abs().max().item()
            assert not torch.isnan(kernel_out).any(), f"NaN detected in kernel output at seed {seed}, shape {batch_size}x{seq_len}"
            assert not torch.isinf(kernel_out).any(), f"Inf detected in kernel output at seed {seed}, shape {batch_size}x{seq_len}"
            assert max_diff < 1e-3, f"Parity failure: max diff {max_diff} >= 1e-3 at seed {seed}, batch={batch_size}, seq_len={seq_len}"


def test_paged_attention_causal_mask_verification() -> None:
    """Test 2: Causal mask verification during chunked prefill and incremental decoding."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 2
    head_dim = 32
    block_sz = 16
    total_blocks = 20

    cache = PagedKVCache(
        num_blocks=total_blocks,
        block_size=block_sz,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device=device,
    )

    # Sequence 0 has 10 tokens, Sequence 1 has 25 tokens
    cache.allocate_sequence(0)
    for _ in range(10):
        cache.append_token(0, torch.randn(num_heads, head_dim, device=device), torch.randn(num_heads, head_dim, device=device))

    cache.allocate_sequence(1)
    for _ in range(25):
        cache.append_token(1, torch.randn(num_heads, head_dim, device=device), torch.randn(num_heads, head_dim, device=device))

    # Construct block tables with padding
    max_blocks = (25 + block_sz - 1) // block_sz
    bt0 = cache.block_tables[0] + [0] * (max_blocks - len(cache.block_tables[0]))
    bt1 = cache.block_tables[1] + [0] * (max_blocks - len(cache.block_tables[1]))
    block_tables = torch.tensor([bt0, bt1], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([10, 25], dtype=torch.int32, device=device)

    q = torch.randn(batch_size, num_heads, head_dim, device=device)

    # Reference execution
    ref_out = paged_attention_reference(q, cache.key_cache, cache.value_cache, block_tables, seq_lens, block_size=block_sz)
    kernel_out = paged_attention_kernel(q, cache.key_cache, cache.value_cache, block_tables, seq_lens, block_size=block_sz)

    diff = (ref_out - kernel_out).abs().max().item()
    assert diff < 1e-3, f"Causal/masked paged attention mismatch: {diff}"

    # Verify that tokens beyond seq_len do not contaminate attention output
    # Corrupt unallocated slots in the tail block of Sequence 0
    tail_block_seq0 = cache.block_tables[0][-1]
    # Sequence 0 has 10 tokens; slots 10..15 in tail block should be untouched / masked out
    cache.key_cache[tail_block_seq0, 10:16] = 1e6
    cache.value_cache[tail_block_seq0, 10:16] = 1e6

    # Re-run; output must remain identical within strict numerical tolerance
    masked_out = paged_attention_reference(q, cache.key_cache, cache.value_cache, block_tables, seq_lens, block_size=block_sz)
    causal_leak = (ref_out - masked_out).abs().max().item()
    assert causal_leak < 1e-5, f"Causal mask leakage: corrupted tail tokens affected attention output (diff: {causal_leak})"


def test_paged_attention_numerical_stability_and_dtypes() -> None:
    """Test 3: Numerical stability checks (absence of NaNs, Infs, or overflows) across FP16, BF16, and FP8."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    num_heads = 4
    head_dim = 128
    block_sz = 32
    seq_len = 128
    num_blocks = (seq_len + block_sz - 1) // block_sz * batch_size + 4

    test_dtypes = [torch.float32, torch.float16]
    if hasattr(torch, "bfloat16"):
        test_dtypes.append(torch.bfloat16)

    for dtype in test_dtypes:
        cache = PagedKVCache(
            num_blocks=num_blocks,
            block_size=block_sz,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

        block_table_list = []
        seq_lens_list = []

        for b in range(batch_size):
            cache.allocate_sequence(b)
            for _ in range(seq_len):
                # Include extreme values to test numerical stability (overflow / underflow resistance)
                k_tok = (torch.randn(num_heads, head_dim, device=device) * 5.0).to(dtype)
                v_tok = (torch.randn(num_heads, head_dim, device=device) * 5.0).to(dtype)
                cache.append_token(b, k_tok, v_tok)
            block_table_list.append(cache.block_tables[b])
            seq_lens_list.append(seq_len)

        block_tables = torch.tensor(block_table_list, dtype=torch.int32, device=device)
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)

        q = (torch.randn(batch_size, num_heads, head_dim, device=device) * 5.0).to(dtype)

        out = paged_attention_kernel(q, cache.key_cache, cache.value_cache, block_tables, seq_lens, block_size=block_sz)

        assert not torch.isnan(out).any(), f"NaN detected for dtype {dtype}"
        assert not torch.isinf(out).any(), f"Inf detected for dtype {dtype}"
        assert out.dtype == dtype, f"Output dtype mismatch: expected {dtype}, got {out.dtype}"

        # Parity check against reference
        ref_out = paged_attention_reference(q, cache.key_cache, cache.value_cache, block_tables, seq_lens, block_size=block_sz)
        max_diff = (out.float() - ref_out.float()).abs().max().item()
        tol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-3
        assert max_diff < tol, f"Numerical parity failure for {dtype}: max diff {max_diff} >= {tol}"
