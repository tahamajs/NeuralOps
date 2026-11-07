#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for vLLM Backend, Custom Ops Dispatcher, and ModelRunner Integration.

Verifies:
- Test 1: Introspection mode returns valid metadata without vLLM installed.
- Test 2: In-tree registration raises IntegrationNotImplementedError when vllm is missing.
- Test 3: Standalone PagedAttention invocation executes cleanly with native Triton / reference backend.
- Test 4: Block table allocation and mapping logic with varying block sizes (16, 32).
- Test 5: SafeTensors authentic weight loading and Anti-Mock compliance.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
import torch.nn as nn

from neuralops.core.exceptions import IntegrationNotImplementedError
import neuralops.integrations.vllm as vllm_integration
from neuralops.integrations.vllm import (
    HAS_VLLM,
    NeuralOpsvLLMMetadata,
    NeuralOpsvLLMImpl,
    NeuralOpsvLLMAttentionBackend,
    NeuralOpsForCausalLMvLLM,
    vllm_is_available,
    register_neuralops_vllm_plugin,
)
from neuralops.integrations.vllm.custom_ops import (
    paged_attention_dispatch,
    paged_attention_custom_op,
    dereference_paged_kv_cache,
    register_vllm_custom_ops,
)


def test_vllm_introspection_metadata_without_vllm():
    """Test 1: Introspection mode returns valid metadata without vLLM installed."""
    assert NeuralOpsvLLMAttentionBackend.get_name() == "NEURALOPS_HYBRID"
    assert NeuralOpsvLLMAttentionBackend.get_impl_cls() is NeuralOpsvLLMImpl
    assert NeuralOpsvLLMAttentionBackend.get_metadata_cls() is NeuralOpsvLLMMetadata

    # Check 4D recurrent state pool shape calculation
    shape_16 = NeuralOpsvLLMAttentionBackend.get_kv_cache_shape(
        num_blocks=64, block_size=16, num_kv_heads=4, head_size=64
    )
    assert shape_16 == (64, 4, 64, 64)

    shape_32 = NeuralOpsvLLMAttentionBackend.get_kv_cache_shape(
        num_blocks=128, block_size=32, num_kv_heads=8, head_size=128
    )
    assert shape_32 == (128, 8, 128, 128)

    # Introspect metadata dataclass without vLLM
    dummy_pt = torch.zeros((2, 4), dtype=torch.long)
    dummy_slots = torch.arange(8, dtype=torch.long)
    dummy_lens = torch.tensor([4, 4], dtype=torch.long)
    meta = NeuralOpsvLLMMetadata(
        num_prefill_tokens=0,
        num_decode_tokens=2,
        seq_ids=[101, 102],
        page_table=dummy_pt,
        slot_mapping=dummy_slots,
        context_lens=dummy_lens,
        is_prompt=False,
        max_context_len=4,
    )
    assert meta.num_decode_tokens == 2
    assert meta.block_tables is dummy_pt
    assert torch.equal(meta.page_table, dummy_pt)


def test_vllm_registration_fail_closed_when_vllm_missing():
    """Test 2: In-tree registration raises IntegrationNotImplementedError when vllm missing."""
    with patch.object(vllm_integration, "HAS_VLLM", False):
        assert vllm_is_available() is False

        impl = NeuralOpsvLLMImpl(num_heads=2, head_size=16, scale=0.25)
        q = torch.randn(1, 1, 32)
        with pytest.raises(IntegrationNotImplementedError) as exc_info:
            impl.forward(q, q, q)
        assert "vLLM is not installed" in str(exc_info.value)
        assert "pip install vllm>=0.6.0" in str(exc_info.value)


def test_standalone_paged_attention_dispatch_and_execution():
    """Test 3: Standalone PagedAttention invocation executes cleanly with native Triton / reference backend."""
    num_blocks = 8
    block_size = 16
    num_heads = 2
    head_size = 32
    batch_size = 2

    k_pool = torch.randn(num_blocks, block_size, num_heads, head_size)
    v_pool = torch.randn(num_blocks, block_size, num_heads, head_size)

    # Sequence lengths: Seq 0 has 20 tokens (requires 2 blocks: 0, 1)
    #                   Seq 1 has 10 tokens (requires 1 block: 2)
    block_tables = torch.tensor([
        [0, 1, -1, -1],
        [2, -1, -1, -1],
    ], dtype=torch.long)
    seq_lens = torch.tensor([20, 10], dtype=torch.long)

    # Decode query: 1 token per sequence [B, 1, H*D]
    q_decode = torch.randn(batch_size, 1, num_heads * head_size)

    # Dispatch should route to Level 2 (Triton) or Level 3 (Vectorized PyTorch)
    out = paged_attention_dispatch(
        q=q_decode,
        k_cache=k_pool,
        v_cache=v_pool,
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
    )

    assert out.shape == q_decode.shape
    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("block_size", [16, 32])
def test_block_table_allocation_and_dereferencing(block_size: int):
    """Test 4: Block table allocation and mapping logic with varying block sizes (16, 32)."""
    batch_size = 3
    num_heads = 4
    head_size = 64
    total_blocks = 16

    k_pool = torch.randn(total_blocks, block_size, num_heads, head_size)
    v_pool = torch.randn(total_blocks, block_size, num_heads, head_size)

    # Allocate sequences:
    # Seq 0: length = block_size + 5 -> 2 blocks (blocks 0, 1)
    # Seq 1: length = block_size * 2 -> 2 blocks (blocks 2, 3)
    # Seq 2: length = 7             -> 1 block  (block 4)
    seq_lens = torch.tensor([block_size + 5, block_size * 2, 7], dtype=torch.long)
    block_tables = torch.tensor([
        [0, 1, -1],
        [2, 3, -1],
        [4, -1, -1],
    ], dtype=torch.long)

    k_seqs, v_seqs = dereference_paged_kv_cache(k_pool, v_pool, block_tables, seq_lens)

    assert len(k_seqs) == batch_size
    assert len(v_seqs) == batch_size

    for b in range(batch_size):
        expected_len = int(seq_lens[b].item())
        assert k_seqs[b].shape == (num_heads, expected_len, head_size)
        assert v_seqs[b].shape == (num_heads, expected_len, head_size)

    # Verify first token of Seq 0 matches block 0 token 0
    # k_pool is [TotalBlocks, BlockSize, NumHeads, HeadDim]
    # k_seqs[0] is [NumHeads, SeqLen, HeadDim]
    expected_k0 = k_pool[0, 0].permute(1, 0)  # [HeadDim, NumHeads] -> wait: k_pool[0, 0] is [NumHeads, HeadDim]
    assert torch.equal(k_seqs[0][:, 0, :], k_pool[0, 0])
    assert torch.equal(v_seqs[0][:, 0, :], v_pool[0, 0])

    # Verify boundary token (at index block_size) comes from block 1 token 0
    assert torch.equal(k_seqs[0][:, block_size, :], k_pool[1, 0])
    assert torch.equal(v_seqs[0][:, block_size, :], v_pool[1, 0])

    # Run custom op with this configuration
    q = torch.randn(batch_size, 1, num_heads, head_size)
    out = paged_attention_custom_op(
        q=q,
        k_pool=k_pool,
        v_pool=v_pool,
        block_tables=block_tables,
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
    )
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_vllm_safetensors_authentic_weights_and_anti_mock():
    """Test 5: Authentic weight loading from SafeTensors complying with Anti-Mock Mandate."""
    from safetensors.torch import save_file
    from neuralops.models import NeuralOpsConfig, NeuralOpsForCausalLM

    cfg = NeuralOpsConfig(
        hidden_size=64,
        num_attention_heads=2,
        num_hidden_layers=1,
        vocab_size=128,
    )

    # 1. Instantiation without weights fails closed
    with pytest.raises(IntegrationNotImplementedError) as exc_info:
        NeuralOpsForCausalLMvLLM(cfg)
    assert "Anti-Mock Mandate" in str(exc_info.value) or "authentic" in str(exc_info.value)

    # 2. Create authentic SafeTensors file
    canonical_model = NeuralOpsForCausalLM(cfg)
    state = canonical_model.state_dict()

    with tempfile.TemporaryDirectory() as tmp_dir:
        st_path = Path(tmp_dir) / "model.safetensors"
        save_file(state, str(st_path))

        # 3. Instantiate with authentic SafeTensors path
        vllm_model = NeuralOpsForCausalLMvLLM(cfg, weights_path=st_path)
        assert vllm_model.model is not None

        # Verify weights match state dict
        for key in list(state.keys())[:5]:
            assert torch.equal(vllm_model.model.state_dict()[key], state[key])

        # Verify forward execution
        input_ids = torch.tensor([[1, 5, 12, 42]], dtype=torch.long)
        logits = vllm_model.forward(input_ids)
        assert logits.shape == (1, 4, 128)
        assert torch.isfinite(logits).all()


def test_chunked_prefill_causal_masking():
    """Verify chunked prefill causal masking in NeuralOpsvLLMImpl."""
    impl = NeuralOpsvLLMImpl(num_heads=2, head_size=16, scale=0.25)
    B, S, H, D = 1, 8, 2, 16
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    v = torch.randn(B, S, H, D)

    # Prefill out with chunk_size = 4
    out = impl._forward_chunked_prefill(q, k, v, chunk_size=4)
    assert out.shape == (B, S, H, D)
    assert torch.isfinite(out).all()

    # Verify causality: modifying future key/value at pos 7 does not change output at pos 2
    k_modified = k.clone()
    k_modified[:, 7, :, :] += 100.0
    out_mod = impl._forward_chunked_prefill(q, k_modified, v, chunk_size=4)
    assert torch.allclose(out[:, :3], out_mod[:, :3], atol=1e-5)
