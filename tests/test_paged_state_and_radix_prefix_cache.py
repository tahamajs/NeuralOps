"""
Unit and Integration Tests for PagedStateTable and RadixPrefixStateCache.
"""

import pytest
import torch

from neuralops.models.paged_state import PagedStateTable, RadixPrefixStateCache
from neuralops.core.triton_kernels.paged_attention import (
    paged_recurrent_state_step_pytorch,
    paged_recurrent_state_step_triton,
)


def test_paged_state_table_allocation_and_free():
    """Verify physical GPU page pool allocation and deallocation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    table = PagedStateTable(num_pages=16, num_heads=4, head_dim=32, device=device)

    assert len(table.free_pages) == 16
    pages_s1 = table.register_sequence(1, initial_pages=2)
    assert len(pages_s1) == 2
    assert len(table.free_pages) == 14

    table.free_sequence(1)
    assert len(table.free_pages) == 16


def test_paged_state_zero_copy_fork_and_cow():
    """Verify O(1) zero-copy sequence forking and Copy-on-Write (CoW)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    table = PagedStateTable(num_pages=16, num_heads=4, head_dim=32, device=device)

    parent_pages = table.register_sequence(10, initial_pages=1)
    p_id = parent_pages[0]
    table.page_pool[p_id, 0, 0, 0] = 42.0

    # Fork sequence 20 from 10
    child_pages = table.fork_sequence(parent_seq_id=10, child_seq_id=20)
    assert child_pages[0] == p_id
    assert table.ref_counts[p_id] == 2

    # Mutate child sequence via prepare_write (CoW)
    new_child_p = table.prepare_write(seq_id=20, page_slot=0)
    assert new_child_p != p_id
    assert table.ref_counts[p_id] == 1
    assert table.ref_counts[new_child_p] == 1
    # Check data was cloned during CoW
    assert float(table.page_pool[new_child_p, 0, 0, 0].item()) == 42.0


def test_radix_prefix_state_cache():
    """Verify prefix matching and associative state caching."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    table = PagedStateTable(num_pages=32, num_heads=4, head_dim=32, device=device)
    cache = RadixPrefixStateCache(table)

    prefix_tokens = [101, 2054, 2003, 1037]  # e.g., "[CLS] What is a"
    page = table.allocate_page()
    table.page_pool[page].fill_(1.23)

    cache.insert_prefix(prefix_tokens, physical_page=page)

    # Exact match
    matched_page, matched_len = cache.match_prefix(prefix_tokens)
    assert matched_page == page
    assert matched_len == 4

    # Longer prompt sharing same prefix
    long_prompt = prefix_tokens + [1045, 2023]
    m_page, m_len = cache.match_prefix(long_prompt)
    assert m_page == page
    assert m_len == 4


def test_paged_recurrent_step_kernel():
    """Verify paged recurrent step output and state mutation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    B, H, D = 2, 4, 32
    table = PagedStateTable(num_pages=8, num_heads=H, head_dim=D, dtype=dtype, device=device)
    table.register_sequence(0, initial_pages=1)
    table.register_sequence(1, initial_pages=1)

    block_tables = table.get_block_table_tensor([0, 1])

    q_t = torch.randn(B, H, D, device=device, dtype=dtype)
    k_t = torch.randn(B, H, D, device=device, dtype=dtype)
    v_t = torch.randn(B, H, D, device=device, dtype=dtype)

    out = paged_recurrent_state_step_triton(
        q_t, k_t, v_t,
        page_pool=table.page_pool,
        block_tables=block_tables,
        page_slot=0,
        decay=0.99,
    )
    assert out.shape == (B, H, D)
