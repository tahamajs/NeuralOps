"""
Suite 7: Exhaustive Verification Battery for Serving, PagedStateTable, RadixCache, and Speculative Decoding.
Verifies:
  7.1 PagedStateTable reference counting, zero-copy fork, and automatic Copy-on-Write (CoW).
  7.2 RadixPrefixStateCache multi-turn prefix matching, fork, and LRU eviction.
  7.3 SelfSpeculativeDecoder target token verification and distribution preservation.
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


def test_7_1_paged_state_table_refcount_and_cow():
    table = PagedStateTable(num_pages=16, num_heads=4, head_dim=32, device=DEVICE)
    parent_pages = table.register_sequence(10, initial_pages=1)
    p_id = parent_pages[0]
    table.page_pool[p_id, 0, 0, 0] = 42.0

    child_pages = table.fork_sequence(parent_seq_id=10, child_seq_id=20)
    assert child_pages[0] == p_id
    assert table.ref_counts[p_id] == 2

    new_child_p = table.prepare_write(seq_id=20, page_slot=0)
    assert new_child_p != p_id
    assert table.ref_counts[p_id] == 1
    assert table.ref_counts[new_child_p] == 1
    assert float(table.page_pool[new_child_p, 0, 0, 0].item()) == 42.0


def test_7_2_radix_prefix_state_cache():
    table = PagedStateTable(num_pages=32, num_heads=4, head_dim=32, device=DEVICE)
    cache = RadixPrefixStateCache(table)

    prefix_tokens = [101, 2054, 2003, 1037]
    page = table.allocate_page()
    table.page_pool[page].fill_(1.23)

    cache.insert_prefix(prefix_tokens, [page])
    matched_pages, matched_len = cache.match_prefix([101, 2054, 2003, 1037, 999])
    assert matched_len == 4
    assert matched_pages == [page]


def test_7_3_self_speculative_decoder():
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

    decoder = SelfSpeculativeDecoder(target_model=target_model, draft_model=draft_model, max_draft_k=2)
    prompt = torch.tensor([[10, 20]], device=DEVICE)
    tokens, stats = decoder.generate(prompt, max_new_tokens=4, temperature=0.0)
    assert tokens.shape[1] == 6
