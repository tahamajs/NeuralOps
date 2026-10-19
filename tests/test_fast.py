#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

import sys
import torch
import torch.nn as nn


from neuralops.inference.paged_cache import PagedKVCache
from neuralops.inference.speculative_decoding import SelfSpeculativeDecoder
from neuralops.models.paged_state import PagedStateTable, RadixPrefixStateCache

def test_fast_inference():
    cache = PagedKVCache(num_blocks=64, block_size=16, num_heads=2, head_dim=16, device="cpu")
    for s in range(35):
        cache.allocate_sequence(s)
        cache.append_token(s, torch.randn(2, 16), torch.randn(2, 16))
    for s in range(35):
        cache.free_sequence(s)

    state_table = PagedStateTable(num_pages=32, num_heads=2, head_dim=16, state_dim=16, device="cpu")
    state_table.register_sequence(1, initial_pages=1)
    state_table.fork_sequence(1, 2)
    p_cow = state_table.prepare_write(2, 0)
    assert p_cow != state_table.seq_page_tables[1][0]
    state_table.free_sequence(1)
    state_table.free_sequence(2)

    radix = RadixPrefixStateCache(state_table)
    page = state_table.allocate_page()
    radix.insert_prefix([1, 2, 3], page)
    mp, ml = radix.match_prefix([1, 2, 3, 4])
    assert ml == 3

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(20, 8)
            self.head = nn.Linear(8, 20)
        def forward(self, ids):
            class Out:
                def __init__(self, l):
                    self.logits = l
            return Out(self.head(self.emb(ids)))

    model = ToyModel()
    model.eval()
    decoder = SelfSpeculativeDecoder(target_model=model, max_draft_k=4, adaptive_k=False)
    ids, stats = decoder.generate(torch.randint(0, 20, (1, 2)), max_new_tokens=4, temperature=0.0)
    assert ids is not None
