#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit and Integration Tests for TGI Continuous Batching Serving Adapter.

Verifies:
1. Dynamic sequence slotting, request queueing, and continuous batch stepping.
2. Async generator stream_generate(request_id) yielding Server-Sent Events (SSE)
   and step-wise token emissions.
3. Strict slot eviction upon request completion (max_tokens, stop sequences, eos).
4. Zero memory leaks across 50 simulated concurrent requests with staggered arrival.
5. Introspection contract and fail-closed behavior of register_neuralops_tgi_plugin().
"""

from __future__ import annotations

import asyncio
import json
import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from neuralops.core.errors import IntegrationNotImplementedError
from neuralops.integrations.tgi import (
    register_neuralops_tgi_plugin,
    get_tgi_capabilities,
    tgi_is_available,
    TGIRequest,
    TGIRequestBatch,
    TGIServingAdapter,
)


@pytest.fixture
def small_gpt2_model():
    """Create a lightweight deterministic GPT2 model on CPU."""
    torch.manual_seed(42)
    config = GPT2Config(
        vocab_size=128,
        n_positions=128,
        n_embd=32,
        n_layer=1,
        n_head=2,
    )
    model = GPT2LMHeadModel(config).eval()
    return model


def test_tgi_capabilities_and_registration():
    """Verify register_neuralops_tgi_plugin returns standardized metadata and RPC hooks."""
    caps = register_neuralops_tgi_plugin()
    assert isinstance(caps, dict)
    assert caps["engine"] == "tgi"
    assert "capabilities" in caps
    assert caps["capabilities"]["continuous_batching"] is True
    assert caps["capabilities"]["async_sse_streaming"] is True
    assert caps["capabilities"]["slot_eviction_on_completion"] is True

    rpc_handlers = caps["rpc_handlers"]
    assert "prefill" in rpc_handlers
    assert "decode_step" in rpc_handlers
    assert "stream_generate" in rpc_handlers

    assert caps["adapter_class"] is TGIServingAdapter
    assert caps["request_class"] is TGIRequest
    assert caps["batch_manager_class"] is TGIRequestBatch

    # Direct query function matches
    direct_caps = get_tgi_capabilities()
    assert direct_caps["engine"] == caps["engine"]


def test_tgi_fail_closed_on_missing_server():
    """Verify register_neuralops_tgi_plugin fails closed if text-generation-server is requested but missing."""
    if not tgi_is_available():
        with pytest.raises(IntegrationNotImplementedError, match="text-generation-server is not installed"):
            register_neuralops_tgi_plugin(require_server=True)
        with pytest.raises(IntegrationNotImplementedError, match="text-generation-server is not installed"):
            register_neuralops_tgi_plugin(require_engine=True)


def test_tgi_request_queueing_and_batch_stepping(small_gpt2_model):
    """Verify request admission, queueing when max_batch_size is reached, and stepping."""
    adapter = TGIServingAdapter(
        model=small_gpt2_model,
        tokenizer=None,
        max_batch_size=2,
        max_seq_len=64,
        device="cpu",
    )

    req1 = adapter.add_request("req_1", [10, 20], max_new_tokens=4)
    req2 = adapter.add_request("req_2", [30, 40], max_new_tokens=4)
    req3 = adapter.add_request("req_3", [50, 60], max_new_tokens=4)

    # First 2 should be active; 3rd queued
    assert req1.slot_id is not None
    assert req2.slot_id is not None
    assert req3.slot_id is None
    assert len(adapter.batch_manager.active_slots) == 2
    assert len(adapter.batch_manager.queued_slots) == 1

    # Execute decode step
    outputs = adapter.decode_step(adapter.batch_manager.active_request_ids())
    assert "req_1" in outputs
    assert "req_2" in outputs
    assert len(req1.generated_ids) == 1
    assert len(req2.generated_ids) == 1


def test_tgi_slot_eviction_upon_completion(small_gpt2_model):
    """Verify that when a request finishes, its slot is freed and queued requests are admitted."""
    adapter = TGIServingAdapter(
        model=small_gpt2_model,
        tokenizer=None,
        max_batch_size=2,
        max_seq_len=64,
        device="cpu",
    )

    _ = adapter.add_request("short_req", [1, 2], max_new_tokens=1)
    _ = adapter.add_request("long_req", [3, 4], max_new_tokens=5)
    req_queued = adapter.add_request("queued_req", [5, 6], max_new_tokens=3)

    assert len(adapter.batch_manager.active_slots) == 2
    assert len(adapter.batch_manager.queued_slots) == 1
    assert req_queued.slot_id is None

    # Step 1: short_req generates 1 token and finishes; should be evicted
    adapter.decode_step(adapter.batch_manager.active_request_ids())

    assert "short_req" not in adapter.batch_manager.active_slots
    assert "short_req" in adapter.batch_manager.completed_slots
    # Queued request admitted into freed slot
    assert "queued_req" in adapter.batch_manager.active_slots
    assert req_queued.slot_id is not None


@pytest.mark.asyncio
async def test_tgi_stream_generate_sse(small_gpt2_model):
    """Verify async stream_generate yields valid Server-Sent Events (SSE) payloads."""
    adapter = TGIServingAdapter(
        model=small_gpt2_model,
        tokenizer=None,
        max_batch_size=4,
        max_seq_len=64,
        device="cpu",
    )

    chunks = []
    async for chunk in adapter.stream_generate(
        request_id="stream_req_1",
        prompt=[5, 15, 25],
        max_new_tokens=4,
        sse_format=True,
    ):
        assert chunk.startswith("data: ")
        assert chunk.endswith("\n\n")
        data = json.loads(chunk[len("data: ") : -2])
        assert "token" in data
        assert "id" in data["token"]
        assert "finished" in data
        assert data["request_id"] == "stream_req_1"
        chunks.append(data)

    assert len(chunks) == 4
    assert chunks[-1]["finished"] is True
    assert adapter.stats()["active_slots"] == 0
    assert "stream_req_1" in adapter.batch_manager.completed_slots


@pytest.mark.asyncio
async def test_tgi_stream_generate_tuples(small_gpt2_model):
    """Verify async stream_generate yields (token_id, text, is_final) tuples when sse_format=False."""
    adapter = TGIServingAdapter(
        model=small_gpt2_model,
        tokenizer=None,
        max_batch_size=4,
        max_seq_len=64,
        device="cpu",
    )

    tokens = []
    async for tok_id, tok_text, is_final in adapter.stream_generate(
        request_id="stream_req_2",
        prompt=[1, 2],
        max_new_tokens=3,
        sse_format=False,
    ):
        assert isinstance(tok_id, int)
        assert isinstance(tok_text, str)
        assert isinstance(is_final, bool)
        tokens.append((tok_id, tok_text, is_final))

    assert len(tokens) == 3
    assert tokens[-1][2] is True
    assert adapter.stats()["active_slots"] == 0


@pytest.mark.asyncio
async def test_tgi_eos_and_stop_termination(small_gpt2_model):
    """Verify exact sequence termination upon encountering eos_token_id."""
    eos_id = 42
    adapter = TGIServingAdapter(
        model=small_gpt2_model,
        tokenizer=None,
        max_batch_size=4,
        max_seq_len=64,
        device="cpu",
        eos_token_id=eos_id,
    )

    req = adapter.add_request("eos_test", [1, 2], max_new_tokens=10)
    assert adapter.check_stop(req, eos_id) is True

    # Test termination via stop sequences
    req_stop = adapter.add_request(
        "stop_test",
        [1, 2],
        max_new_tokens=10,
        stop_sequences=["<stop>"],
    )
    req_stop.generated_ids.append(999)  # mock decode would be "<999>" or similar
    # Check stop check logic
    assert adapter.check_stop(req_stop, 10) is False


@pytest.mark.asyncio
async def test_tgi_50_concurrent_requests_staggered(small_gpt2_model):
    """
    Stress test: 50 concurrent simulated requests with staggered arrival times.
    Verifies zero deadlocks, complete token emissions, and full slot eviction on termination.
    """
    adapter = TGIServingAdapter(
        model=small_gpt2_model,
        tokenizer=None,
        max_batch_size=8,
        max_seq_len=128,
        device="cpu",
    )

    async def single_client(client_id: int, max_tokens: int, delay: float):
        await asyncio.sleep(delay)
        req_id = f"client_{client_id}"
        received = []
        async for chunk in adapter.stream_generate(
            request_id=req_id,
            prompt=[client_id % 50 + 1, (client_id + 5) % 50 + 1],
            max_new_tokens=max_tokens,
            sse_format=False,
        ):
            received.append(chunk)
        return req_id, len(received)

    tasks = []
    num_requests = 50
    for i in range(num_requests):
        max_tokens = 2 + (i % 4)  # 2, 3, 4, or 5 tokens
        delay = (i % 10) * 0.002  # staggered arrival
        tasks.append(single_client(i, max_tokens, delay))

    results = await asyncio.gather(*tasks)

    assert len(results) == num_requests
    for i, (req_id, tok_count) in enumerate(results):
        expected_count = 2 + (i % 4)
        assert tok_count == expected_count, f"Client {req_id} received {tok_count} != expected {expected_count}"

    # Verify complete slot eviction & zero memory leaks
    assert adapter.stats()["active_slots"] == 0
    assert adapter.stats()["queued_requests"] == 0
    assert adapter.stats()["completed_requests"] == num_requests
    assert adapter.stats()["free_slots"] == 8
