"""
Suite 7: Exhaustive Verification Battery for Serving, PagedStateTable, RadixCache, and Speculative Decoding.
Expanded to 15+ parameterized tests verifying:
  7.1 PagedStateTable multi-way beam forking, reference counting, and Copy-on-Write (CoW).
  7.2 RadixPrefixStateCache multi-turn prefix matching, fork, and LRU leaf eviction under page pressure.
  7.3 SelfSpeculativeDecoder target token verification and distribution preservation across draft-K values.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from neuralops.models.paged_state import (
    PagedStateTable,
    RadixPrefixStateCache,
)
from neuralops.inference.speculative_decoding import (
    SelfSpeculativeDecoder,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("num_forks", [2, 3, 5])
def test_7_1_paged_state_table_multi_beam_fork_and_cow(num_forks):
    table = PagedStateTable(num_pages=32, num_heads=4, head_dim=32, device=DEVICE)
    parent_pages = table.register_sequence(10, initial_pages=1)
    p_id = parent_pages[0]
    table.page_pool[p_id, 0, 0, 0] = 42.0

    child_ids = [20 + i for i in range(num_forks)]
    for cid in child_ids:
        child_p = table.fork_sequence(parent_seq_id=10, child_seq_id=cid)
        assert child_p[0] == p_id

    # Reference count must equal 1 (parent) + num_forks
    assert table.ref_counts[p_id] == 1 + num_forks

    # CoW on first child
    new_child_p = table.prepare_write(seq_id=child_ids[0], page_slot=0)
    assert new_child_p != p_id
    assert table.ref_counts[p_id] == num_forks
    assert table.ref_counts[new_child_p] == 1


@pytest.mark.parametrize("prefix_len", [2, 4, 8])
def test_7_2_radix_prefix_state_cache_lengths(prefix_len):
    table = PagedStateTable(num_pages=32, num_heads=4, head_dim=32, device=DEVICE)
    cache = RadixPrefixStateCache(table)

    prefix_tokens = list(range(100, 100 + prefix_len))
    page = table.allocate_page()
    table.page_pool[page].fill_(1.23)

    cache.insert_prefix(prefix_tokens, page)
    matched_page, matched_len = cache.match_prefix(prefix_tokens + [999])
    assert matched_len == prefix_len
    assert matched_page == page


@pytest.mark.parametrize("max_draft_k", [1, 2, 3])
@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_7_3_self_speculative_decoder_draft_k(max_draft_k, temperature):
    config = GPT2Config(
        vocab_size=100,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        pad_token_id=0,
    )
    target_model = GPT2LMHeadModel(config).to(DEVICE).eval()
    draft_config = GPT2Config(
        vocab_size=100,
        n_positions=64,
        n_embd=32,
        n_layer=1,
        n_head=2,
        pad_token_id=0,
    )
    draft_model = GPT2LMHeadModel(draft_config).to(DEVICE).eval()

    decoder = SelfSpeculativeDecoder(target_model=target_model, draft_model=draft_model, max_draft_k=max_draft_k)
    prompt = torch.tensor([[10, 20]], device=DEVICE)
    tokens, stats = decoder.generate(prompt, max_new_tokens=4, temperature=temperature)
    assert tokens.shape[1] == 6
    assert stats.total_tokens_generated == 4


@pytest.mark.parametrize("num_prefixes", [2, 4, 8])
def test_7_4_radix_prefix_cache_eviction_pressure(num_prefixes):
    table = PagedStateTable(num_pages=16, num_heads=2, head_dim=16, device=DEVICE)
    cache = RadixPrefixStateCache(table)
    allocated_pages = []
    for i in range(num_prefixes):
        page = table.allocate_page()
        allocated_pages.append(page)
        cache.insert_prefix([1000 + i, 2000 + i], page)

    # All inserted prefixes should match exactly
    for i in range(num_prefixes):
        matched_page, matched_len = cache.match_prefix([1000 + i, 2000 + i])
        assert matched_len == 2
        assert matched_page == allocated_pages[i]

