#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Serving & Runtime Resilience Test Suite.

Verifies:
1. Continuous batching concurrency safety, dynamic sequence slotting, and admission control.
2. Paged KV-cache block allocation, Copy-on-Write (CoW), zero memory leaks, and defragmentation.
3. AutonomousCircuitBreaker three-state FSM transitions (Normal -> Degraded -> Tripped -> Auto-Recovery).
4. Multi-threaded and multi-task asynchronous serving request queuing without race conditions.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from typing import Any, List

import pytest
import torch
import torch.nn as nn

from neuralops.runtime.circuit_breaker import (
    AutonomousCircuitBreaker,
    CircuitBreakerFSM,
    CircuitState,
)
from neuralops.runtime.config import RuntimeConfig
from neuralops.runtime.context import get_runtime_config, use_runtime
from neuralops.runtime.events import EventSink, RunEvent
from neuralops.serving.paged_cache_manager import (
    PagedKVCache,
    PagedKVCacheManager,
)
from neuralops.serving.request_batcher import (
    TGIRequest,
    TGIRequestBatch,
    TGIServingAdapter,
)


class MockResilientModel(nn.Module):
    """Deterministic mock model for verifying decode stepping and batching invariants."""

    def __init__(self, vocab_size: int = 128):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Any:
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        logits = torch.zeros((batch_size, seq_len, self.vocab_size), dtype=torch.float32)

        # Emit next token deterministically as (last_token + 1) % vocab_size
        last_tok = input_ids[:, -1]
        for b in range(batch_size):
            next_val = int((int(last_tok[b].item()) + 1) % self.vocab_size)
            logits[b, -1, next_val] = 10.0

        if use_cache:
            # Simulated lightweight past_kv structure: tuple of dummy tensors
            dummy_kv = (torch.zeros((batch_size, 1, 1, 1)), torch.zeros((batch_size, 1, 1, 1)))
            return type("Outputs", (), {"logits": logits, "past_key_values": dummy_kv})()
        return logits


# =========================================================================
# 1. Paged KV-Cache Zero-Leak & Allocation Invariants
# =========================================================================

def test_paged_kv_cache_initial_invariants():
    """Verify clean initialization state of PagedKVCache."""
    cache = PagedKVCache(
        num_blocks=32,
        block_size=16,
        num_heads=4,
        head_dim=32,
        dtype=torch.float32,
        device="cpu",
    )
    assert cache.num_blocks == 32
    assert len(cache.free_blocks) == 32
    assert len(cache.block_tables) == 0
    assert len(cache.block_ref_counts) == 0
    assert cache.key_cache.shape == (32, 16, 4, 32)
    assert cache.value_cache.shape == (32, 16, 4, 32)


def test_paged_kv_cache_zero_memory_leak_lifecycle():
    """Verify zero memory leak over repetitive sequence allocation, expansion, and deletion."""
    cache = PagedKVCache(
        num_blocks=64,
        block_size=8,
        num_heads=2,
        head_dim=16,
        dtype=torch.float32,
        device="cpu",
    )
    initial_free = len(cache.free_blocks)
    assert initial_free == 64

    # Run 50 cycles of allocating 10 sequences, filling multiple blocks, and freeing them
    for cycle in range(50):
        seq_ids = [1000 + i for i in range(10)]
        for seq_id in seq_ids:
            cache.allocate_sequence(seq_id)
            # Append 25 tokens per sequence -> consumes ceil(25 / 8) = 4 blocks per seq = 40 blocks total
            keys = torch.randn(25, 2, 16)
            vals = torch.randn(25, 2, 16)
            cache.append_tokens(seq_id, keys, vals)
            assert cache.seq_lengths[seq_id] == 25
            assert len(cache.block_tables[seq_id]) == 4

        assert len(cache.free_blocks) == 64 - 40
        assert sum(cache.block_ref_counts.values()) == 40

        # Free all sequences
        for seq_id in seq_ids:
            cache.free_sequence(seq_id)

        # Invariant: Complete reclamation of free blocks and reference counts
        assert len(cache.free_blocks) == initial_free
        assert len(cache.block_tables) == 0
        assert len(cache.block_ref_counts) == 0
        assert len(cache.seq_lengths) == 0


def test_paged_kv_cache_cow_forking_and_mutation():
    """Verify Copy-on-Write forking logic and physical block isolation upon mutation."""
    cache = PagedKVCache(
        num_blocks=16,
        block_size=4,
        num_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )

    parent_id = 1
    cache.allocate_sequence(parent_id)

    # Append 4 tokens to completely fill 1 block (block 0)
    k_init = torch.ones(4, 2, 8)
    v_init = torch.ones(4, 2, 8)
    cache.append_tokens(parent_id, k_init, v_init)

    parent_blocks = cache.get_block_table(parent_id)
    assert len(parent_blocks) == 1
    shared_block = parent_blocks[0]
    assert cache.block_ref_counts[shared_block] == 1

    # Fork child from parent (e.g. beam search / tree sampling branch)
    child_id = 2
    cache.fork_sequence(parent_id, child_id)
    assert cache.get_block_table(child_id) == [shared_block]
    assert cache.block_ref_counts[shared_block] == 2
    assert len(cache.free_blocks) == 15

    # Mutate child sequence by appending a token; triggers CoW because shared block is not full (wait, shared block is full, so new block is allocated)
    # Let's test non-full block mutation CoW:
    cache2 = PagedKVCache(
        num_blocks=16,
        block_size=4,
        num_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )
    cache2.allocate_sequence(10)
    # Append 2 tokens into block of size 4
    cache2.append_tokens(10, torch.full((2, 2, 8), 1.0), torch.full((2, 2, 8), 1.0))
    blk = cache2.get_block_table(10)[0]
    assert cache2.block_ref_counts[blk] == 1

    # Fork sequence 20 from 10
    cache2.fork_sequence(10, 20)
    assert cache2.block_ref_counts[blk] == 2

    # Append 1 token to sequence 20 -> token_offset is 2 != 0, and ref_count > 1 -> CoW triggers!
    tok_k = torch.full((2, 8), 7.0)
    tok_v = torch.full((2, 8), 7.0)
    cache2.append_token(20, tok_k, tok_v)

    # Check that sequence 20 now points to a distinct physical block
    blk_parent = cache2.get_block_table(10)[0]
    blk_child = cache2.get_block_table(20)[0]
    assert blk_parent != blk_child
    assert cache2.block_ref_counts[blk_parent] == 1
    assert cache2.block_ref_counts[blk_child] == 1

    # Ensure parent's data was NOT overwritten
    k_parent, v_parent = cache2.get_kv(10)
    assert torch.all(k_parent == 1.0)
    assert k_parent.shape == (2, 2, 8)

    # Ensure child's data has both initial tokens and the new token
    k_child, v_child = cache2.get_kv(20)
    assert k_child.shape == (3, 2, 8)
    assert torch.all(k_child[:2] == 1.0)
    assert torch.all(k_child[2] == 7.0)


def test_paged_kv_cache_defragmentation_and_compaction():
    """Verify defragmentation consolidates scattered blocks into contiguous virtual sequences."""
    cache = PagedKVCache(
        num_blocks=16,
        block_size=4,
        num_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )

    # Allocate sequences 1, 2, 3
    for s in (1, 2, 3):
        cache.allocate_sequence(s)
        cache.append_tokens(s, torch.ones(8, 2, 8) * s, torch.ones(8, 2, 8) * s)

    # Sequences 1, 2, 3 occupy blocks [0, 1], [2, 3], [4, 5]
    # Free sequence 2 to create a hole in physical allocation
    cache.free_sequence(2)
    assert cache.free_blocks == [2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # Run defragmentation
    stats = cache.defragment()
    assert stats["compacted_blocks"] == 4  # 2 for seq 1, 2 for seq 3
    assert stats["reclaimed_blocks"] == 12

    # Check that remaining sequences 1 and 3 are compacted contiguously
    assert cache.get_block_table(1) == [0, 1]
    assert cache.get_block_table(3) == [2, 3]

    # Verify content integrity
    k1, _ = cache.get_kv(1)
    k3, _ = cache.get_kv(3)
    assert torch.all(k1 == 1.0)
    assert torch.all(k3 == 3.0)


# =========================================================================
# 2. Continuous Batching Concurrency & Request Slotting Safety
# =========================================================================

def test_request_batch_dynamic_slotting_and_eviction():
    """Verify strict slot binding, queue admission, and eviction in TGIRequestBatch."""
    batch = TGIRequestBatch(max_batch_size=3, max_seq_len=64)

    req1 = TGIRequest("r1", torch.tensor([1, 2, 3], dtype=torch.long), max_new_tokens=4)
    req2 = TGIRequest("r2", torch.tensor([4, 5], dtype=torch.long), max_new_tokens=4)
    req3 = TGIRequest("r3", torch.tensor([6, 7, 8, 9], dtype=torch.long), max_new_tokens=4)
    req4 = TGIRequest("r4", torch.tensor([10], dtype=torch.long), max_new_tokens=4)

    # Admit first 3 requests -> occupy slots 0, 1, 2
    s1 = batch.admit_request(req1)
    s2 = batch.admit_request(req2)
    s3 = batch.admit_request(req3)
    assert s1 == 0 and s2 == 1 and s3 == 2
    assert len(batch.active_slots) == 3
    assert len(batch.free_slot_ids) == 0

    # 4th request must be queued
    s4 = batch.admit_request(req4)
    assert s4 is None
    assert len(batch.queued_slots) == 1

    # Mark r2 as finished and evict
    req2.finished = True
    evicted = batch.evict_completed_slots()
    assert evicted == ["r2"]

    # r4 should automatically be promoted into the freed slot (slot 1)
    assert "r4" in batch.active_slots
    assert batch.active_slots["r4"].slot_id == 1
    assert len(batch.queued_slots) == 0


def test_request_batch_concurrency_race_free():
    """Verify that multi-threaded concurrent additions and removals do not corrupt slots."""
    batch = TGIRequestBatch(max_batch_size=16, max_seq_len=128)
    num_requests = 40

    def worker_submit(idx: int):
        req = TGIRequest(f"req_{idx}", torch.tensor([idx % 50 + 1], dtype=torch.long), max_new_tokens=5)
        try:
            batch.admit_request(req)
            return True
        except Exception:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker_submit, i) for i in range(num_requests)]
        results = [f.result() for f in futures]

    assert all(results)
    assert len(batch.active_slots) + len(batch.queued_slots) == num_requests
    assert len(batch.active_slots) <= batch.max_batch_size

    # Validate slot IDs are uniquely mapped
    assigned_slots = [req.slot_id for req in batch.active_slots.values()]
    assert len(assigned_slots) == len(set(assigned_slots)), "Slot collision detected!"


def test_tgi_serving_adapter_sequential_truthful_generation():
    """Verify TGIServingAdapter end-to-end decode stepping without silent crashes."""
    model = MockResilientModel(vocab_size=100)
    adapter = TGIServingAdapter(
        model=model,
        tokenizer=None,
        max_batch_size=4,
        max_seq_len=64,
        device="cpu",
    )

    prompts = [
        torch.tensor([1, 2, 3], dtype=torch.long),
        torch.tensor([10, 20], dtype=torch.long),
    ]

    completions = adapter.generate_sequential(prompts, max_new_tokens=3)
    assert len(completions) == 2
    # Verify stats
    stats = adapter.stats()
    assert stats["total_requests"] >= 2
    assert stats["total_tokens_generated"] >= 6


@pytest.mark.asyncio
async def test_async_stream_generation_sse():
    """Verify asynchronous SSE token streaming iterator."""
    model = MockResilientModel(vocab_size=100)
    adapter = TGIServingAdapter(
        model=model,
        tokenizer=None,
        max_batch_size=4,
        max_seq_len=64,
        device="cpu",
    )

    chunks: List[str] = []
    async for sse_chunk in adapter.stream_generate(
        prompt=torch.tensor([5, 6], dtype=torch.long),
        max_new_tokens=3,
        sse_format=True,
    ):
        chunks.append(sse_chunk)

    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk.startswith("data: {")
        assert chunk.endswith("\n\n")


# =========================================================================
# 3. Autonomous Circuit Breaker Three-State FSM Transitions
# =========================================================================

def test_circuit_breaker_fsm_state_transitions():
    """Verify deterministic state transitions: Normal -> Degraded -> Tripped."""
    fsm = CircuitBreakerFSM(cooldown_seconds=0.050, failure_threshold=1)
    assert fsm.state == CircuitState.NORMAL

    # Record first failure -> moves to DEGRADED
    state1 = fsm.record_failure("Kernel Panic 1")
    assert state1 == CircuitState.DEGRADED
    assert fsm.state == CircuitState.DEGRADED

    # Record second failure -> moves to TRIPPED
    state2 = fsm.record_failure("Kernel Panic 2")
    assert state2 == CircuitState.TRIPPED
    assert fsm.state == CircuitState.TRIPPED
    assert fsm.total_trips == 1


def test_circuit_breaker_auto_recovery_cooldown():
    """Verify autonomous cooldown recovery back to NORMAL after cooldown expires."""
    # Fast 10ms cooldown for test execution
    fsm = CircuitBreakerFSM(cooldown_seconds=0.010, failure_threshold=1)
    fsm.record_failure("Failure 1")  # DEGRADED
    fsm.record_failure("Failure 2")  # TRIPPED
    assert fsm.state == CircuitState.TRIPPED

    # Before cooldown elapses, check_recovery stays TRIPPED
    assert fsm.check_recovery() == CircuitState.TRIPPED

    # Wait 25ms (> 2 * cooldown)
    time.sleep(0.025)

    recovered = fsm.check_recovery()
    assert recovered == CircuitState.NORMAL
    assert fsm.state == CircuitState.NORMAL


def test_autonomous_circuit_breaker_fallback_execution():
    """Verify AutonomousCircuitBreaker invokes tier 2 and tier 3 upon failures."""
    acb = AutonomousCircuitBreaker(cooldown_seconds=0.020, enable_oom_healing=True)

    tier1_called = 0
    tier2_called = 0
    tier3_called = 0

    def l1_fn(x: torch.Tensor) -> torch.Tensor:
        nonlocal tier1_called
        tier1_called += 1
        raise RuntimeError("Synthetic Level 1 CUDA C++ kernel panic")

    def l2_fn(x: torch.Tensor) -> torch.Tensor:
        nonlocal tier2_called
        tier2_called += 1
        raise RuntimeError("Synthetic Level 2 Triton JIT panic")

    def l3_fn(x: torch.Tensor) -> torch.Tensor:
        nonlocal tier3_called
        tier3_called += 1
        return x * 2.0  # Safe PyTorch reference

    x = torch.tensor([1.0, 2.0, 3.0])

    # First call: Level 1 fails -> record failure -> degrades -> Level 2 fails -> trips -> Level 3 succeeds!
    res = acb.execute(l1_fn, l2_fn, l3_fn, x)
    assert torch.allclose(res, x * 2.0)
    assert tier1_called == 1
    assert tier2_called == 1
    assert tier3_called == 1
    assert acb.fsm.state == CircuitState.TRIPPED

    # Second call immediately: already TRIPPED, so directly routes to tier 3
    res2 = acb.execute(l1_fn, l2_fn, l3_fn, x)
    assert torch.allclose(res2, x * 2.0)
    assert tier3_called == 2
    assert tier1_called == 1


def test_circuit_breaker_fail_closed_contract():
    """Verify that if all 3 tiers fail, a structured RuntimeError is raised without silent catch."""
    acb = AutonomousCircuitBreaker(cooldown_seconds=0.020)

    def err_fn(x: torch.Tensor) -> torch.Tensor:
        raise ValueError("Fatal computation error")

    with pytest.raises(RuntimeError) as exc_info:
        acb.execute(err_fn, err_fn, err_fn, torch.zeros(1))

    assert "All circuit-breaker levels failed" in str(exc_info.value)


# =========================================================================
# 4. Runtime Context & Configuration Isolation
# =========================================================================

def test_runtime_context_scoping_and_isolation():
    """Verify ContextVar-based runtime context isolation with use_runtime."""
    base_cfg = get_runtime_config()
    assert base_cfg.device in ("auto", "cpu", "cuda", "mps")

    with use_runtime(device="cpu", dtype="float32", seed=123):
        scoped_cfg = get_runtime_config()
        assert scoped_cfg.device == "cpu"
        assert scoped_cfg.dtype == "float32"
        assert scoped_cfg.seed == 123
        assert scoped_cfg.resolve_device() == torch.device("cpu")
        assert scoped_cfg.resolve_dtype() == torch.float32

    # Verify restoration of previous config
    restored_cfg = get_runtime_config()
    assert restored_cfg.seed == base_cfg.seed


def test_runtime_events_decoupled_telemetry():
    """Verify thread-safe EventSink pub-sub mechanisms."""
    sink = EventSink()
    received_events: List[RunEvent] = []

    def on_event(ev: RunEvent) -> None:
        received_events.append(ev)

    sink.subscribe("serving.batch.admit", on_event)
    ev = sink.emit("serving.batch.admit", batch_size=4, slot_id=0)

    assert len(received_events) == 1
    assert received_events[0].topic == "serving.batch.admit"
    assert received_events[0].payload["batch_size"] == 4
    assert received_events[0].payload["slot_id"] == 0
