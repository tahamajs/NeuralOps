# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Incremental & Robust Stress Test Suite for NeuralOps v1.4 (Agent Gamma)
"""

import sys
print("Starting python test script...", flush=True)

import torch
import torch.nn as nn
print(f"PyTorch version: {torch.__version__}", flush=True)

from neuralops.inference.paged_cache import PagedKVCache
from neuralops.inference.speculative_decoding import SelfSpeculativeDecoder
from neuralops.models.paged_state import PagedStateTable, RadixPrefixStateCache

print("Successfully imported NeuralOps modules.", flush=True)

def test_1_paged_kv_cache():
    print("[Test 1] PagedKVCache continuous batching (32+ sequences)...", flush=True)
    cache = PagedKVCache(num_blocks=64, block_size=16, num_heads=2, head_dim=16, device="cpu")
    for seq_id in range(35):
        cache.allocate_sequence(seq_id)
        cache.append_token(seq_id, torch.randn(2, 16), torch.randn(2, 16))
    for seq_id in range(35):
        k, v = cache.get_kv(seq_id)
        assert k.shape[0] == 1
    for seq_id in range(35):
        cache.free_sequence(seq_id)
    print("[Test 1] PASSED", flush=True)

def test_2_paged_state_and_cow():
    print("[Test 2] PagedStateTable & Copy-on-Write...", flush=True)
    state_table = PagedStateTable(num_pages=32, num_heads=2, head_dim=16, state_dim=16, device="cpu")
    parent = 1
    child = 2
    state_table.register_sequence(parent, initial_pages=1)
    state_table.page_pool[state_table.seq_page_tables[parent][0]].fill_(5.0)
    
    state_table.fork_sequence(parent, child)
    p_cow = state_table.prepare_write(child, page_slot=0)
    state_table.page_pool[p_cow].fill_(10.0)
    
    assert torch.allclose(state_table.page_pool[state_table.seq_page_tables[parent][0]], torch.tensor(5.0))
    assert torch.allclose(state_table.page_pool[p_cow], torch.tensor(10.0))
    
    state_table.free_sequence(parent)
    state_table.free_sequence(child)
    print("[Test 2] PASSED", flush=True)

def test_3_radix_cache():
    print("[Test 3] RadixPrefixStateCache...", flush=True)
    state_table = PagedStateTable(num_pages=16, num_heads=2, head_dim=16, state_dim=16, device="cpu")
    radix = RadixPrefixStateCache(state_table)
    page = state_table.allocate_page()
    radix.insert_prefix([1, 2, 3], page)
    matched_page, matched_len = radix.match_prefix([1, 2, 3, 4])
    assert matched_page == page
    assert matched_len == 3
    print("[Test 3] PASSED", flush=True)

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(16, 32)
    def forward(self, x):
        return torch.zeros((x.shape[0], x.shape[1], 32))

class ToyEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 16)
        self.head = nn.Linear(16, 100)
    def forward(self, input_ids):
        x = self.emb(input_ids)
        logits = self.head(x)
        class Out:
            def __init__(self, l):
                self.logits = l
        return Out(logits)

def test_4_speculative_decoding():
    print("[Test 4] SelfSpeculativeDecoder gamma sweep & divergence...", flush=True)
    model = ToyEmbeddingModel()
    model.eval()
    decoder = SelfSpeculativeDecoder(target_model=model, max_draft_k=4, adaptive_k=False)
    input_ids = torch.randint(0, 100, (1, 4))
    
    for gamma in [1, 2, 4]:
        decoder.max_draft_k = gamma
        decoder.current_k = gamma
        ids, stats = decoder.generate(input_ids, max_new_tokens=8, temperature=0.0)
        assert ids.shape[-1] == 4 + 8
        print(f"   Gamma {gamma} passed. Tokens/sec: {stats.tokens_per_sec}")
    print("[Test 4] PASSED", flush=True)

if __name__ == "__main__":
    test_1_paged_kv_cache()
    test_2_paged_state_and_cow()
    test_3_radix_cache()
    test_4_speculative_decoding()
    print("ALL TESTS COMPLETED SUCCESSFULLY!", flush=True)
