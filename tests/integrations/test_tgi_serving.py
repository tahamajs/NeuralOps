#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Integration and Continuous Batching Tests for TGI (Text Generation Inference).

Verifies:
- Test 1: Capabilities dictionary export and RPC hook bindings.
- Test 2: Dynamic slot allocation, sequence progression, and eviction on EOS.
- Test 3: Async SSE streaming iterator yields expected tokens.
- Test 4: Concurrency stress-test: multiple overlapping requests with staggered arrivals.
"""

from __future__ import annotations

import asyncio
import json
import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.integrations.tgi import (
    register_neuralops_tgi_plugin,
    get_tgi_capabilities,
    create_tgi_app,
    tgi_is_available,
    TGIRequest,
    TGIRequestBatch,
    TGIServingAdapter,
)


@pytest.fixture
def small_lm():
    """Create a deterministic miniature GPT2 language model on CPU."""
    torch.manual_seed(1337)
    config = GPT2Config(
        vocab_size=256,
        n_positions=128,
        n_embd=32,
        n_layer=1,
        n_head=2,
    )
    return GPT2LMHeadModel(config).eval()


def test_tgi_capabilities_and_rpc_hooks():
    """Test 1: Capabilities dictionary export and RPC hook bindings."""
    caps = get_tgi_capabilities()
    assert isinstance(caps, dict)
    assert caps["engine"] == "tgi"
    assert "runtime_bridge_version" in caps
    assert "capabilities" in caps

    capabilities = caps["capabilities"]
    assert capabilities["continuous_batching"] is True
    assert capabilities["dynamic_sequence_slotting"] is True
    assert capabilities["async_sse_streaming"] is True
    assert capabilities["step_wise_generation"] is True
    assert capabilities["slot_eviction_on_completion"] is True
    assert capabilities["left_padding_assembly"] is True

    rpc_handlers = caps["rpc_handlers"]
    assert "prefill" in rpc_handlers
    assert "decode_step" in rpc_handlers
    assert "stream_generate" in rpc_handlers

    assert caps["adapter_class"] is TGIServingAdapter
    assert caps["request_class"] is TGIRequest
    assert caps["batch_manager_class"] is TGIRequestBatch

    # Verify registration function introspection returns the same capabilities
    reg_caps = register_neuralops_tgi_plugin()
    assert reg_caps["engine"] == "tgi"
    assert reg_caps["capabilities"] == capabilities

    # Verify fail-closed when server is required but absent
    if not tgi_is_available():
        with pytest.raises(IntegrationNotImplementedError, match="text-generation-server is not installed"):
            register_neuralops_tgi_plugin(require_server=True)


def test_tgi_dynamic_slot_allocation_progression_and_eos_eviction(small_lm):
    """Test 2: Dynamic slot allocation, sequence progression, and eviction on EOS."""
    max_batch_size = 2
    batch = TGIRequestBatch(max_batch_size=max_batch_size, max_seq_len=64)

    # 1. Dynamic sequence slotting: fixed pre-allocated batch buffer of size B_max
    assert len(batch.free_slot_ids) == max_batch_size
    assert len(batch.active_slots) == 0

    eos_token_id = 9
    req1 = TGIRequest("req_1", torch.tensor([1, 2, 3]), max_new_tokens=10, eos_token_id=eos_token_id)
    req2 = TGIRequest("req_2", torch.tensor([4, 5]), max_new_tokens=10, eos_token_id=eos_token_id)
    req3 = TGIRequest("req_3", torch.tensor([6, 7, 8]), max_new_tokens=10, eos_token_id=eos_token_id)

    # Admit requests
    slot1 = batch.admit_request(req1)
    slot2 = batch.admit_request(req2)
    slot3 = batch.admit_request(req3)

    assert slot1 == 0
    assert slot2 == 1
    assert slot3 is None  # Queued because max_batch_size is 2
    assert len(batch.active_slots) == 2
    assert len(batch.queued_slots) == 1
    assert len(batch.free_slot_ids) == 0

    # Test left-padded batch assembly
    batch_dict = batch.assemble_batch(pad_token_id=0, padding_side="left")
    assert batch_dict["input_ids"].shape == (2, 3)
    assert batch_dict["attention_mask"].shape == (2, 3)
    # req2 had len 2, padded to len 3 on the left
    assert batch_dict["input_ids"][1, 0].item() == 0
    assert batch_dict["attention_mask"][1, 0].item() == 0
    assert batch_dict["attention_mask"][1, 1].item() == 1

    # Simulate EOS on req1
    req1.generated_ids.append(eos_token_id)
    req1.finished = True
    req1.finish_reason = "eos"

    # Evict completed slots
    evicted = batch.evict_completed_slots()
    assert "req_1" in evicted
    assert "req_1" not in batch.active_slots
    assert "req_1" in batch.completed_slots

    # req3 should be admitted into the freed slot
    assert "req_3" in batch.active_slots
    assert req3.slot_id == slot1
    assert len(batch.queued_slots) == 0

    # Test batch.step(model)
    step_results = batch.step(small_lm)
    assert "req_2" in step_results
    assert "req_3" in step_results
    assert len(req2.generated_ids) == 1
    assert len(req3.generated_ids) == 1


def test_tgi_async_sse_streaming_iterator(small_lm):
    """Test 3: Async SSE streaming iterator yields expected tokens."""
    adapter = TGIServingAdapter(
        model=small_lm,
        tokenizer=None,
        max_batch_size=4,
        max_seq_len=64,
        device="cpu",
    )

    async def run_sse():
        prompt = "Hello TGI"
        max_new_tokens = 4
        chunks = []
        async for sse_chunk in adapter.stream_generate(prompt=prompt, max_new_tokens=max_new_tokens, sse_format=True):
            assert isinstance(sse_chunk, str)
            assert sse_chunk.startswith("data: ")
            assert sse_chunk.endswith("\n\n")
            data = json.loads(sse_chunk[len("data: ") : -2])
            assert "token" in data
            assert "finished" in data
            chunks.append(data)
        return chunks

    chunks = asyncio.run(run_sse())
    assert len(chunks) == 4
    assert chunks[-1]["finished"] is True

    # 2. Test stream_generate with tuple format
    async def run_tuples():
        tuple_chunks = []
        async for tok_id, tok_text, is_final in adapter.stream_generate(
            request_id="test_req_tuple",
            prompt=[1, 2, 3],
            max_new_tokens=3,
            sse_format=False,
        ):
            assert isinstance(tok_id, int)
            assert isinstance(tok_text, str)
            assert isinstance(is_final, bool)
            tuple_chunks.append((tok_id, tok_text, is_final))
        return tuple_chunks

    tuple_chunks = asyncio.run(run_tuples())
    assert len(tuple_chunks) == 3
    assert tuple_chunks[-1][2] is True

    # 3. Test standalone HTTP/SSE FastAPI hook
    app = create_tgi_app(adapter)
    assert app is not None
    assert hasattr(app, "routes")
    route_paths = [route.path for route in app.routes]
    assert "/health" in route_paths
    assert "/info" in route_paths
    assert "/generate" in route_paths
    assert "/generate_stream" in route_paths

    # Verify zero slot leakage
    assert adapter.stats()["active_slots"] == 0


def test_tgi_concurrency_stress_staggered_arrivals(small_lm):
    """Test 4: Concurrency stress-test: multiple overlapping requests with staggered arrivals."""
    max_batch_size = 4
    adapter = TGIServingAdapter(
        model=small_lm,
        tokenizer=None,
        max_batch_size=max_batch_size,
        max_seq_len=128,
        device="cpu",
    )

    total_requests = 24

    async def client_request(idx: int, delay: float, num_tokens: int):
        await asyncio.sleep(delay)
        req_id = f"client_{idx}"
        received_tokens = []
        async for chunk in adapter.stream_generate(
            request_id=req_id,
            prompt=[(idx * 3 + j) % 60 + 1 for j in range(3)],
            max_new_tokens=num_tokens,
            sse_format=False,
        ):
            received_tokens.append(chunk)
        return req_id, len(received_tokens)

    async def run_all():
        tasks = []
        for i in range(total_requests):
            delay = (i % 6) * 0.003
            num_tokens = 2 + (i % 5)
            tasks.append(client_request(i, delay, num_tokens))
        return await asyncio.gather(*tasks)

    results = asyncio.run(run_all())
    assert len(results) == total_requests

    for i, (req_id, count) in enumerate(results):
        expected = 2 + (i % 5)
        assert count == expected, f"Request {req_id} got {count} tokens, expected {expected}"

    # Verify zero memory leaks and complete slot eviction
    stats = adapter.stats()
    assert stats["active_slots"] == 0
    assert stats["queued_requests"] == 0
    assert stats["completed_requests"] == total_requests
    assert stats["free_slots"] == max_batch_size
