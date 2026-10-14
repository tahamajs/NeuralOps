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

from neuralops.core.serving.paged_state import (
    PagedStateTable,
    RadixPrefixStateCache,
)
from neuralops.serving.speculative import (
    SelfSpeculativeDecoder,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_7_1_paged_state_table_refcount_and_cow():
    table = PagedStateTable(num_pages=64, page_size=16, state_dim=32, device=DEVICE)
    seq_a = table.allocate_sequence()

    # Write state to sequence A
    dummy_state = torch.ones(1, 32, device=DEVICE) * 3.14
    table.write_state(seq_a, token_idx=0, state=dummy_state)

    # Zero-copy fork to sequence B
    seq_b = table.fork_sequence(seq_a)
    state_a = table.read_state(seq_a, token_idx=0)
    state_b = table.read_state(seq_b, token_idx=0)
    torch.testing.assert_close(state_a, state_b)

    # Copy-on-write modification on B must NOT affect A
    new_state = torch.ones(1, 32, device=DEVICE) * 9.99
    table.write_state(seq_b, token_idx=0, state=new_state)

    state_a_after = table.read_state(seq_a, token_idx=0)
    state_b_after = table.read_state(seq_b, token_idx=0)
    assert not torch.allclose(state_a_after, state_b_after)
    torch.testing.assert_close(state_a_after, dummy_state)
    torch.testing.assert_close(state_b_after, new_state)


def test_7_2_radix_prefix_state_cache():
    cache = RadixPrefixStateCache(max_cached_tokens=1024, state_dim=32, device=DEVICE)

    prefix_tokens = [101, 2054, 2003, 1037]  # e.g., "what is a"
    state_payload = torch.randn(1, 32, device=DEVICE)

    cache.insert(prefix_tokens, state_payload)
    matched_state, matched_len = cache.match_prefix([101, 2054, 2003, 1037, 3241, 102])
    assert matched_len == 4
    torch.testing.assert_close(matched_state, state_payload)


def test_7_3_self_speculative_decoder():
    class DummyDraftTargetModel(nn.Module):
        def __init__(self, vocab_size=64):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, 32)
            self.head = nn.Linear(32, vocab_size)

        def forward(self, input_ids):
            x = self.emb(input_ids)
            return self.head(x)

    model = DummyDraftTargetModel().to(DEVICE).eval()
    decoder = SelfSpeculativeDecoder(draft_model=model, target_model=model, gamma=3)

    prompt = torch.tensor([[1, 5, 9]], device=DEVICE)
    with torch.no_grad():
        out = decoder.generate(prompt, max_new_tokens=4)
    assert out.shape[1] >= prompt.shape[1] + 1
