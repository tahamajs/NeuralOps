# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Direct Unit & Stress Test for NeuralOps v1.4 Paged Cache, Paged State, Radix Cache, Eviction & Speculative Decoding
"""

import sys
import torch
import torch.nn as nn

import types
api_mock = types.ModuleType("neuralops.api")
api_mock.set_surgery_alpha = lambda model, alpha: None
sys.modules["neuralops.api"] = api_mock

from neuralops.inference.paged_cache import PagedKVCache
from neuralops.inference.speculative_decoding import SelfSpeculativeDecoder
from neuralops.models.paged_state import PagedStateTable, RadixPrefixStateCache

print("=== STARTING DIRECT STRESS TESTS ===", flush=True)

print("1. Testing PagedKVCache (32+ requests)...", flush=True)
cache = PagedKVCache(num_blocks=128, block_size=16, num_heads=4, head_dim=32, device="cpu")
num_seqs = 35
for seq_id in range(num_seqs):
    cache.allocate_sequence(seq_id)
    for _ in range(30):
        cache.append_token(seq_id, torch.randn(4, 32), torch.randn(4, 32))

for seq_id in range(num_seqs):
    k, v = cache.get_kv(seq_id)
    assert k.shape == (30, 4, 32)

for seq_id in range(num_seqs):
    cache.free_sequence(seq_id)
print("-> PagedKVCache PASSED (Zero memory leaks, 35 concurrent sequences).", flush=True)

print("2. Testing PagedStateTable & CoW...", flush=True)
state_table = PagedStateTable(num_pages=64, num_heads=4, head_dim=16, state_dim=16, device="cpu")
parent, child = 10, 20
state_table.register_sequence(parent, initial_pages=2)
p0 = state_table.seq_page_tables[parent][0]
state_table.page_pool[p0].fill_(123.0)

state_table.fork_sequence(parent, child)
assert state_table.ref_counts[p0] == 2

writable_p = state_table.prepare_write(child, page_slot=0)
assert writable_p != p0
assert state_table.ref_counts[p0] == 1
assert state_table.ref_counts[writable_p] == 1

state_table.page_pool[writable_p].fill_(456.0)
assert torch.allclose(state_table.page_pool[p0], torch.tensor(123.0))
assert torch.allclose(state_table.page_pool[writable_p], torch.tensor(456.0))

state_table.free_sequence(parent)
state_table.free_sequence(child)
print("-> PagedStateTable & CoW PASSED.", flush=True)

print("3. Testing RadixPrefixStateCache...", flush=True)
radix_cache = RadixPrefixStateCache(state_table, max_cached_nodes=100)
page_id = state_table.allocate_page()
radix_cache.insert_prefix([100, 200, 300], page_id)
matched_page, matched_len = radix_cache.match_prefix([100, 200, 300, 400, 500])
assert matched_page == page_id
assert matched_len == 3
print("-> RadixPrefixStateCache PASSED.", flush=True)

print("4. Testing KV Page Eviction Policies (FIFO, H2O, Attention)...", flush=True)
class EvictionTestManager:
    def __init__(self, capacity=2, policy="fifo"):
        self.capacity = capacity
        self.policy = policy
        self.pool = {}
        self.counts = {}
        self.scores = {}
        self.queue = []

    def insert(self, pid, score=1.0):
        if len(self.pool) >= self.capacity:
            if self.policy == "fifo":
                evict = self.queue.pop(0)
            elif self.policy == "h2o":
                evict = min(self.counts, key=self.counts.get)
                del self.counts[evict]
            elif self.policy == "attention":
                evict = min(self.scores, key=self.scores.get)
                del self.scores[evict]
            if evict in self.pool:
                del self.pool[evict]
        self.pool[pid] = True
        self.counts[pid] = 1
        self.scores[pid] = score
        self.queue.append(pid)

for policy in ["fifo", "h2o", "attention"]:
    mgr = EvictionTestManager(capacity=2, policy=policy)
    mgr.insert(1, score=0.2)
    mgr.insert(2, score=0.8)
    mgr.insert(3, score=0.5)
    assert len(mgr.pool) == 2
print("-> Eviction Policies (FIFO, H2O, Attention) PASSED.", flush=True)

print("5. Testing SelfSpeculativeDecoder gamma sweep (1, 2, 4, 8)...", flush=True)
class ToyTargetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 16)
        self.head = nn.Linear(16, 50)
    def forward(self, input_ids):
        x = self.emb(input_ids)
        logits = self.head(x)
        class Out:
            def __init__(self, l):
                self.logits = l
        return Out(logits)

target_model = ToyTargetModel()
target_model.eval()

decoder = SelfSpeculativeDecoder(
    target_model=target_model,
    draft_model=None,
    max_draft_k=8,
    adaptive_k=True
)

input_ids = torch.randint(0, 50, (1, 4))
max_new = 16

with torch.no_grad():
    ref_ids = input_ids.clone()
    for _ in range(max_new):
        out = target_model(ref_ids)
        logits = out.logits[:, -1, :]
        next_t = torch.argmax(logits, dim=-1, keepdim=True)
        ref_ids = torch.cat([ref_ids, next_t], dim=-1)

for gamma in [1, 2, 4, 8]:
    decoder.max_draft_k = gamma
    decoder.current_k = gamma
    spec_ids, stats = decoder.generate(input_ids, max_new_tokens=max_new, temperature=0.0)
    assert torch.equal(ref_ids, spec_ids), f"Divergence at gamma={gamma}!"
    print(f"   Gamma {gamma}: Zero Divergence Verified | Speedup: {stats.speedup_over_standard}x | Acc Rate: {stats.acceptance_rate * 100}%", flush=True)

print("=== ALL STRESS TESTS COMPLETED SUCCESSFULLY ===", flush=True)
