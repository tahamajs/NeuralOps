#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Regression test suite for surgery, serving, and inference pipeline debugging.
Verifies AST rewriting across architectures, LoRA/DoRA adapter preservation,
paged KV-cache zero-copy view consistency, fragmentation telemetry, and continuous batching slotting.
"""

import pytest
import torch
import torch.nn as nn

from neuralops.inference.paged_cache import PagedKVCache
from neuralops.serving.request_batcher import TGIRequest, TGIRequestBatch, TGIServingAdapter
from neuralops.surgery.endoskeletal_shell import UniversalAttentionBridge
from neuralops.surgery.scanner import UniversalArchitectureScanner


class MockMistralAttention(nn.Module):
    """Simulates a Mistral attention block without explicit is_causal attribute."""

    def __init__(self, d: int = 64, heads: int = 4):
        super().__init__()
        self.num_heads = heads
        self.num_key_value_heads = heads
        self.head_dim = d // heads
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, hidden_states, **kwargs):
        return self.o_proj(hidden_states), None


class MockGemmaAttention(nn.Module):
    """Simulates a Gemma attention block without explicit is_causal attribute."""

    def __init__(self, d: int = 64, heads: int = 4, head_dim: int = 16):
        super().__init__()
        self.num_heads = heads
        self.num_key_value_heads = heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d, heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d, heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, d, bias=False)

    def forward(self, hidden_states, **kwargs):
        return self.o_proj(hidden_states), None


def test_scanner_causal_detection_mistral_and_gemma():
    """Verify scanner infers is_causal=True for Mistral and Gemma architectures when attribute is omitted."""
    mistral_mod = MockMistralAttention()
    mistral_specs = UniversalArchitectureScanner.scan(mistral_mod)
    assert len(mistral_specs) == 1
    assert mistral_specs[0].is_causal is True

    gemma_mod = MockGemmaAttention()
    gemma_specs = UniversalArchitectureScanner.scan(gemma_mod)
    assert len(gemma_specs) == 1
    assert gemma_specs[0].is_causal is True


def test_scanner_dora_magnitude_normalization():
    """Verify scanner applies DoRA magnitude vector normalization correctly."""
    linear = nn.Linear(32, 64, bias=False)
    # Mock DoRA attributes on the linear projection
    linear.lora_A = {"default": nn.Parameter(torch.randn(8, 32))}
    linear.lora_B = {"default": nn.Parameter(torch.randn(64, 8))}
    linear.scaling = {"default": 2.0}
    linear.active_adapter = "default"
    # Magnitude vector for 64 output channels
    linear.lora_magnitude_vector = {"default": nn.Parameter(torch.ones(64) * 5.0)}

    w, b = UniversalArchitectureScanner._extract_weight_bias(linear)
    assert w.shape == (64, 32)
    # Check that row norms of extracted weights match magnitude vector (5.0)
    row_norms = torch.linalg.norm(w, dim=1)
    assert torch.allclose(row_norms, torch.full((64,), 5.0), atol=1e-4)


def test_bridge_preserves_dora_adapters():
    """Verify UniversalAttentionBridge extracts and preserves DoRA magnitude vectors."""
    host = MockMistralAttention(d=32, heads=2)
    host.q_proj.lora_A = nn.Parameter(torch.randn(4, 32))
    host.q_proj.lora_B = nn.Parameter(torch.randn(32, 4))
    host.q_proj.scaling = 1.0
    host.q_proj.lora_magnitude_vector = nn.Parameter(torch.ones(32))

    specs = UniversalArchitectureScanner.scan(host)
    spec = specs[0]

    bridge = UniversalAttentionBridge(
        orig_module=host,
        spec=spec,
        new_attention=nn.Linear(32, 32),
        inherit_weights=False,
    )

    adapters = bridge.get_lora_adapters()
    assert "q_proj_lora_magnitude_vector" in adapters
    assert bridge.has_lora_adapters is True


def test_bridge_split_4proj_cache_unpacking():
    """Verify UniversalAttentionBridge returns a 3-tuple when use_cache or past_key_value is provided."""
    host = MockMistralAttention(d=32, heads=2)
    specs = UniversalArchitectureScanner.scan(host)
    spec = specs[0]

    student = nn.Linear(32, 32)
    bridge = UniversalAttentionBridge(
        orig_module=host,
        spec=spec,
        new_attention=student,
        inherit_weights=False,
    )

    x = torch.randn(2, 4, 32)
    # When use_cache=True, HF decoder layers unpack (hidden, weights, present_key_value)
    out = bridge(x, use_cache=True)
    assert isinstance(out, tuple)
    assert len(out) == 3
    assert out[0].shape == (2, 4, 32)
    assert out[1] is None
    assert out[2] is None


def test_paged_cache_zero_copy_blocks_and_telemetry_hygiene():
    """Verify get_kv_blocks returns zero-copy view for contiguous allocations and telemetry ignores 0-refs."""
    cache = PagedKVCache(num_blocks=8, block_size=4, num_heads=2, head_dim=8)
    cache.allocate_sequence(1)

    for i in range(8):
        cache.append_token(1, torch.full((2, 8), float(i)), torch.full((2, 8), float(i * 10)))

    # Both blocks 0 and 1 are contiguous
    k_blocks, v_blocks = cache.get_kv_blocks(1)
    assert k_blocks.shape == (2, 4, 2, 8)
    # Check that data pointer matches key_cache storage directly (zero-copy view)
    assert k_blocks.data_ptr() == cache.key_cache[0].data_ptr()

    # Querying a non-existent block via defaultdict should not inflate telemetry
    _ = cache.block_ref_counts[7]
    stats = cache.get_fragmentation_stats()
    assert stats["allocated_blocks"] == 2  # Only blocks 0 and 1 are active


def test_paged_cache_empty_defragment():
    """Verify defragmenting an empty cache zeroes data and reports clean post_stats."""
    cache = PagedKVCache(num_blocks=4, block_size=2, num_heads=1, head_dim=2)
    res = cache.defragment()
    assert res["compacted_blocks"] == 0
    assert res["reclaimed_blocks"] == 4
    assert res["post_stats"]["allocated_blocks"] == 0


def test_serving_adapter_empty_generated_ids_dtype():
    """Verify TGIServingAdapter does not cause dtype promotion to float when generated_ids is empty."""
    class EchoModel(nn.Module):
        def forward(self, input_ids, **kwargs):
            # Enforce that input_ids is strictly integer dtype
            if input_ids.dtype not in (torch.int64, torch.int32):
                raise TypeError(f"Expected integer token IDs, got {input_ids.dtype}")
            b, s = input_ids.shape
            return torch.zeros(b, s, 50, device=input_ids.device)

    model = EchoModel()
    adapter = TGIServingAdapter(model=model, max_batch_size=2, device="cpu")
    req = adapter.add_request("r1", prompt=[1, 2, 3], max_new_tokens=2)

    # Decode step without prefill triggers the fallback path with generated_ids=[]
    # This must not throw TypeError due to float promotion
    tokens = adapter.decode_step(["r1"])
    assert "r1" in tokens
