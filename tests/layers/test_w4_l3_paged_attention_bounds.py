#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for Paged Attention block table bounds checking."""

import pytest
import torch

from neuralops.layers.paged_attention import (
    PagedAttention,
    paged_attention_decode,
    validate_block_tables,
)


def test_paged_attention_valid_table_passes():
    """Verify that a valid block table passes validation and executes cleanly."""
    b, h, d = 2, 4, 32
    block_size = 16
    num_cache_blocks = 10

    q = torch.randn(b, h, d)
    k_cache = torch.randn(num_cache_blocks, block_size, h, d)
    v_cache = torch.randn(num_cache_blocks, block_size, h, d)

    seq_lens = torch.tensor([12, 28], dtype=torch.long)
    # Batch 0 needs 1 block; Batch 1 needs 2 blocks
    block_tables = torch.tensor([[0, 2, 0, 0], [1, 3, 0, 0]], dtype=torch.long)

    # Must pass validation without raising
    validate_block_tables(block_tables, num_cache_blocks, seq_lens, block_size=block_size)

    out = paged_attention_decode(q, k_cache, v_cache, block_tables, seq_lens, block_size=block_size)
    assert out.shape == (b, h, d)
    assert torch.isfinite(out).all()


def test_paged_attention_out_of_bounds_block_id_raises():
    """Verify that block tables containing out-of-bounds page IDs raise ValueError."""
    num_cache_blocks = 8  # Valid block indices are 0..7
    block_size = 16

    # Case 1: Page ID >= num_cache_blocks
    seq_lens = torch.tensor([10], dtype=torch.long)
    bad_table_upper = torch.tensor([[8]], dtype=torch.long)  # 8 is out of bounds
    with pytest.raises(ValueError, match="Out-of-bounds page ID"):
        validate_block_tables(bad_table_upper, num_cache_blocks, seq_lens, block_size=block_size)

    # Case 2: Negative page ID
    bad_table_neg = torch.tensor([[-1]], dtype=torch.long)
    with pytest.raises(ValueError, match="Out-of-bounds page ID"):
        validate_block_tables(bad_table_neg, num_cache_blocks, seq_lens, block_size=block_size)


def test_paged_attention_layer_forward():
    """Test PagedAttention layer forward pass."""
    b, h, d = 2, 4, 32
    block_size = 16
    num_cache_blocks = 12

    layer = PagedAttention(dim=h * d, num_heads=h, head_dim=d, block_size=block_size)
    x = torch.randn(b, 1, h * d)
    k_cache = torch.randn(num_cache_blocks, block_size, h, d)
    v_cache = torch.randn(num_cache_blocks, block_size, h, d)

    seq_lens = torch.tensor([14, 25], dtype=torch.long)
    block_tables = torch.tensor([[0, 1, 0, 0], [2, 3, 0, 0]], dtype=torch.long)

    out, _ = layer(x, k_cache, v_cache, block_tables, seq_lens)
    assert out.shape == (b, 1, h * d)
    assert torch.isfinite(out).all()
