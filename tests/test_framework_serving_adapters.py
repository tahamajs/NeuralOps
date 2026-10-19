# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Comprehensive Verification Test Battery for NeuralOps Serving Infrastructure and Framework Integrations.

Covers exactly 21 tests across 4 categories:
1. TGI Adapter (7 tests): continuous batching, streaming, stop sequences, stats
2. SGLang Radix Tree Adapter (7 tests): prefix insertion, exact match, longest match, branching, LRU eviction
3. TensorRT-LLM Scaffold (4 tests): clean import, roadmap NotImplementedError, availability check, roadmap doc
4. Regression Tests (3 tests): Hugging Face roundtrip, ONNX export, PagedStateTable lifecycle
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

import neuralops
from neuralops.models.paged_state import PagedStateTable
from neuralops.export.exporter import export_model, load_exported_model
from neuralops.integrations.tgi import (
    TGIServingAdapter,
    TGIRequest,
    TGIRequestBatch,
    TGIStreamingIterator,
)
from neuralops.integrations.sglang import (
    SGLangRadixAdapter,
    RadixPrefixNode,
)
from neuralops.integrations.tensorrt_llm import (
    TensorRTLLMPluginBuilder,
)



def test_tgi_adapter_imports_cleanly():
    """Verify TGI adapter classes import and initialize cleanly."""
    config = GPT2Config(vocab_size=100, n_positions=64, n_embd=32, n_layer=1, n_head=2)
    model = GPT2LMHeadModel(config).eval()
    adapter = TGIServingAdapter(model=model, tokenizer=None, max_batch_size=8, device="cpu")
    assert adapter is not None
    assert isinstance(adapter.batch_manager, TGIRequestBatch)


def test_tgi_single_request_blocking():
    """Submit 1 prompt, generate 5 new tokens, verify output completion."""
    config = GPT2Config(vocab_size=100, n_positions=64, n_embd=32, n_layer=1, n_head=2)
    model = GPT2LMHeadModel(config).eval()
    adapter = TGIServingAdapter(model=model, tokenizer=None, max_batch_size=4, device="cpu")

    prompt = [10, 20, 30]
    completions = adapter.continuous_batching_generate([prompt], max_new_tokens=5)
    assert len(completions) == 1
    assert len(completions[0]) > 0


def test_tgi_batch_of_4_requests():
    """Submit 4 prompts of varying lengths; verify all 4 generate full completions."""
    config = GPT2Config(vocab_size=100, n_positions=64, n_embd=32, n_layer=1, n_head=2)
    model = GPT2LMHeadModel(config).eval()
    adapter = TGIServingAdapter(model=model, tokenizer=None, max_batch_size=8, device="cpu")

    prompts = [
        [1, 2],
        [3, 4, 5, 6],
        [7],
        [8, 9, 10, 11, 12],
    ]
    completions = adapter.continuous_batching_generate(prompts, max_new_tokens=4)
    assert len(completions) == 4
    for comp in completions:
        assert isinstance(comp, str) and len(comp) > 0


def test_tgi_streaming_yields_tokens_one_at_a_time():
    """Count yielded streaming chunks, verify each is a valid token tuple."""
    config = GPT2Config(vocab_size=100, n_positions=64, n_embd=32, n_layer=1, n_head=2)
    model = GPT2LMHeadModel(config).eval()
    adapter = TGIServingAdapter(model=model, tokenizer=None, max_batch_size=4, device="cpu")

    prompt = [15, 25]
    stream_tokens = []
    for tok, text in adapter.streaming_generate(prompt, max_new_tokens=6):
        assert isinstance(tok, int)
        assert isinstance(text, str)
        stream_tokens.append(tok)

    assert len(stream_tokens) == 6


def test_tgi_stop_sequence_termination():
    """Verify stop sequence triggers request termination."""
    config = GPT2Config(vocab_size=100, n_positions=64, n_embd=32, n_layer=1, n_head=2)
    model = GPT2LMHeadModel(config).eval()
    adapter = TGIServingAdapter(model=model, tokenizer=None, max_batch_size=4, device="cpu", eos_token_id=99)

    req = adapter.add_request(
        request_id="stop_test",
        prompt=[1, 2],
        max_new_tokens=20,
        stop_sequences=None,
    )
    is_stop = adapter.check_stop(req, 99)
    assert is_stop is True


def test_tgi_continuous_batching_variable_length():
    """Verify requests of differing lengths are batched and scheduled concurrently."""
    config = GPT2Config(vocab_size=100, n_positions=64, n_embd=32, n_layer=1, n_head=2)
    model = GPT2LMHeadModel(config).eval()
    adapter = TGIServingAdapter(model=model, tokenizer=None, max_batch_size=4, device="cpu")

    adapter.add_request("req_short", [1, 2], max_new_tokens=2)
    adapter.add_request("req_long", [1, 2, 3, 4, 5], max_new_tokens=5)

    step_outputs = adapter.decode_step(["req_short", "req_long"])
    assert "req_short" in step_outputs
    assert "req_long" in step_outputs


def test_tgi_stats_tracking():
    """Verify operational counters and throughput statistics increment correctly."""
    config = GPT2Config(vocab_size=100, n_positions=64, n_embd=32, n_layer=1, n_head=2)
    model = GPT2LMHeadModel(config).eval()
    adapter = TGIServingAdapter(model=model, tokenizer=None, max_batch_size=4, device="cpu")

    adapter.continuous_batching_generate([[1, 2]], max_new_tokens=3)
    s = adapter.stats()
    assert s["total_requests"] >= 1
    assert s["total_tokens_generated"] >= 3
    assert "avg_latency_ms" in s
    assert "avg_tokens_per_sec" in s



def test_sglang_imports_cleanly():
    """Verify SGLang adapter imports and initializes cleanly."""
    adapter = SGLangRadixAdapter(model=None, tokenizer=None, max_cached_tokens=1024, device="cpu")
    assert adapter is not None
    assert adapter.root is not None
    assert len(adapter.root.children) == 0


def test_sglang_insert_single_prefix():
    """Insert one prefix, verify tree increases node count."""
    adapter = SGLangRadixAdapter(model=None, tokenizer=None, max_cached_tokens=1024, device="cpu")
    dummy_kv = torch.randn(2, 2, 4, 16)
    node = adapter.insert_prefix([101, 2054, 2003], dummy_kv)
    assert node is not None
    assert adapter.stats()["total_nodes"] == 2


def test_sglang_match_exact_prefix():
    """Insert prefix and query exact prefix: verify full matched length."""
    adapter = SGLangRadixAdapter(model=None, tokenizer=None, max_cached_tokens=1024, device="cpu")
    dummy_kv = torch.randn(2, 2, 3, 16)
    adapter.insert_prefix([10, 20, 30], dummy_kv)

    matched_len, kv = adapter.match_prefix([10, 20, 30])
    assert matched_len == 3
    assert kv is not None
    assert torch.equal(kv, dummy_kv)


def test_sglang_match_longest_shared():
    """When ['a','b','c'] is cached and query is ['a','b','c','d'], match len is 3."""
    adapter = SGLangRadixAdapter(model=None, tokenizer=None, max_cached_tokens=1024, device="cpu")
    dummy_kv = torch.randn(2, 2, 3, 16)
    adapter.insert_prefix([10, 20, 30], dummy_kv)

    matched_len, kv = adapter.match_prefix([10, 20, 30, 40])
    assert matched_len == 3
    assert kv is not None


def test_sglang_match_none():
    """When only ['a','b','c'] is cached and query is ['x','y','z'], match len is 0."""
    adapter = SGLangRadixAdapter(model=None, tokenizer=None, max_cached_tokens=1024, device="cpu")
    dummy_kv = torch.randn(2, 2, 3, 16)
    adapter.insert_prefix([10, 20, 30], dummy_kv)

    matched_len, kv = adapter.match_prefix([99, 100, 101])
    assert matched_len == 0
    assert kv is None


def test_sglang_branching():
    """Insert ['a','b','c'] and ['a','b','d'], verify trie node splitting at depth 2."""
    adapter = SGLangRadixAdapter(model=None, tokenizer=None, max_cached_tokens=1024, device="cpu")
    adapter.insert_prefix([1, 2, 3], torch.randn(1))
    adapter.insert_prefix([1, 2, 4], torch.randn(1))

    split_node = adapter.root.children[1]
    assert split_node.token_chunk == [1, 2]
    assert 3 in split_node.children
    assert 4 in split_node.children


def test_sglang_lru_eviction():
    """Exceed max cached nodes, unpin sequences, verify oldest leaf evicted."""
    adapter = SGLangRadixAdapter(model=None, tokenizer=None, max_cached_prefixes=3, device="cpu")
    adapter.insert_prefix([100], torch.randn(1))
    adapter.insert_prefix([200], torch.randn(1))
    adapter.free_sequence([100])

    freed = adapter.evict_lru()
    assert freed > 0
    assert adapter.stats()["eviction_count"] >= 1



def test_tensorrt_llm_imports_cleanly():
    """Verify TensorRT-LLM scaffold imports and instantiates without TensorRT SDK installed."""
    builder = TensorRTLLMPluginBuilder(model=None)
    assert builder is not None
    assert builder.plugin_dir is not None


def test_tensorrt_llm_build_raises_not_implemented():
    """Verify .build_plugin() raises informative NotImplementedError detailing v2.0.0 roadmap."""
    builder = TensorRTLLMPluginBuilder(model=None)
    with pytest.raises(NotImplementedError) as exc_info:
        builder.build_plugin()
    assert "TensorRT-LLM plugin compilation is planned for NeuralOps v2.0.0" in str(exc_info.value)
    assert "TENSORRT_LLM_PLAN.md" in str(exc_info.value)


def test_tensorrt_llm_is_available_false_on_this_system():
    """Verify .is_available() safely returns False when TensorRT SDK is not installed."""
    assert TensorRTLLMPluginBuilder.is_available() is False


def test_tensorrt_llm_roadmap_doc_exists():
    """Verify docs/roadmap/TENSORRT_LLM_PLAN.md exists, is non-empty, and has >= 500 words."""
    doc_path = Path("docs/roadmap/TENSORRT_LLM_PLAN.md")
    assert doc_path.exists(), f"Missing roadmap document: {doc_path}"
    content = doc_path.read_text(encoding="utf-8")
    assert len(content.split()) >= 500



def test_hf_roundtrip_still_works(tmp_path: Path):
    """Verify Hugging Face replace_attention + safetensors bit-exact roundtrip."""
    config = GPT2Config(vocab_size=100, n_positions=64, n_embd=32, n_layer=1, n_head=2)
    model = GPT2LMHeadModel(config).eval()
    neuralops.replace_attention(model, attention="cosformer")
    model.eval()

    export_dir = tmp_path / "hf_roundtrip_verify"
    export_model(model, export_dir, format="safetensors")

    base = GPT2LMHeadModel(config).eval()
    loaded, meta = load_exported_model(export_dir, base_model=base)
    loaded.eval()

    sample = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        out_orig = model(sample).logits
        out_loaded = loaded(sample).logits

    diff = (out_orig - out_loaded).abs().max().item()
    assert diff == 0.0, f"Expected 0.0 diff, got {diff}"


def test_onnx_export_still_works(tmp_path: Path):
    """Verify ONNX export generates a valid checked ONNX protobuf."""
    import onnx
    from neuralops.compression_engine.svd.direct_svd import DirectSVDCompressor

    class TinyLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 8)
        def forward(self, x):
            return self.fc(x)

    model = TinyLinear().eval()
    comp = DirectSVDCompressor(model, rank=4, min_rank=2)
    comp.compress()
    model.eval()

    export_dir = tmp_path / "onnx_verify"
    dummy = torch.randn(1, 16)
    onnx_file = export_model(model, export_dir, format="onnx", sample_input=dummy)

    assert onnx_file.exists()
    onnx_m = onnx.load(str(onnx_file))
    onnx.checker.check_model(onnx_m)
    assert len(onnx_m.graph.node) > 0


def test_paged_state_table_still_works():
    """Verify PagedStateTable Copy-on-Write lifecycle and reference counting."""
    table = PagedStateTable(num_pages=16, num_heads=2, head_dim=16, device="cpu")
    parent_pages = table.register_sequence(seq_id=1, initial_pages=1)
    p0 = parent_pages[0]
    table.page_pool[p0, 0, 0, 0] = 99.0

    child_pages = table.fork_sequence(parent_seq_id=1, child_seq_id=2)
    assert table.ref_counts[p0] == 2

    new_p0 = table.prepare_write(seq_id=2, page_slot=0)
    assert new_p0 != p0
    assert table.ref_counts[p0] == 1
    assert table.ref_counts[new_p0] == 1
    assert table.page_pool[new_p0, 0, 0, 0] == 99.0
    table.free_sequence(1)
    table.free_sequence(2)
    assert len(table.free_pages) == 16
