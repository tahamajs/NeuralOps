#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for vLLM Serving Adapter (Fail-Closed Contract & Backend Harmonization).
"""

from unittest.mock import patch
import pytest
import torch

from neuralops.core.errors import IntegrationNotImplementedError
import neuralops.integrations.vllm as vllm_integration
from neuralops.integrations.vllm import (
    HAS_VLLM,
    NeuralOpsAttentionBackend,
    NeuralOpsRecurrentSpec,
    PagedStateTable,
    VLLMAdapter,
)
from neuralops.integrations.vllm_backend.backend import (
    NeuralOpsAttentionBackend as BackendBackend,
)


def test_vllm_adapter_fail_closed_or_minimal_generation():
    """Verify VLLMAdapter fails closed with IntegrationNotImplementedError when vllm is missing."""
    if not HAS_VLLM:
        with pytest.raises(IntegrationNotImplementedError) as exc_info:
            _ = VLLMAdapter()
        assert "vLLM is not installed" in str(exc_info.value)
    else:
        adapter = VLLMAdapter()
        assert adapter is not None


def test_neuralops_attention_backend_name_and_kv_cache_shape():
    """Verify NeuralOpsAttentionBackend identity and 4D recurrent state pool shape."""
    assert NeuralOpsAttentionBackend.get_name() == "NEURALOPS"
    shape = NeuralOpsAttentionBackend.get_kv_cache_shape(8, 16, 4, 32)
    assert shape == (8, 4, 32, 32)
    assert NeuralOpsAttentionBackend is BackendBackend
    assert NeuralOpsRecurrentSpec is not None
    assert PagedStateTable is not None


def test_neuralops_vllm_impl_initialization_and_fail_closed():
    """Verify NeuralOpsvLLMImpl initialization and fail-closed checks when vLLM is not installed."""
    impl = vllm_integration.NeuralOpsvLLMImpl(
        num_heads=4,
        head_size=32,
        scale=0.125,
        num_kv_heads=4,
        block_size=16,
    )
    assert impl.num_heads == 4
    assert impl.head_size == 32
    assert impl.scale == 0.125
    assert impl.block_size == 16

    q = torch.randn(2, 4, 32)
    k = torch.randn(2, 4, 32)
    v = torch.randn(2, 4, 32)

    # When HAS_VLLM is False, forward() must fail closed with IntegrationNotImplementedError
    with patch.object(vllm_integration, "HAS_VLLM", False):
        with pytest.raises(IntegrationNotImplementedError) as exc_info:
            impl.forward(q, k, v)
        assert "vLLM is not installed" in str(exc_info.value)
        assert "vllm>=0.6.0" in str(exc_info.value)


def test_neuralops_vllm_impl_forward_dispatch():
    """Verify NeuralOpsvLLMImpl forward slot dispatch and reference execution when vLLM is present."""
    impl = vllm_integration.NeuralOpsvLLMImpl(
        num_heads=2,
        head_size=16,
        scale=1.0 / 4.0,
        num_kv_heads=2,
        block_size=16,
    )

    B, T, D = 2, 8, 32
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)

    class DummyMetadata:
        slot_mapping = torch.arange(B * T, dtype=torch.long)
        page_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        context_lens = torch.tensor([T, T], dtype=torch.long)

    with patch.object(vllm_integration, "HAS_VLLM", True):
        # 1. Clean reference execution when kv_cache is None
        out_ref = impl.forward(q, k, v, kv_cache=None, attn_metadata=None)
        assert out_ref.shape == (B, T, D)

        # 2. Slot mapping dispatch with kv_cache
        num_blocks = 8
        k_cache = torch.randn(num_blocks, 16, 2, 16)
        v_cache = torch.randn(num_blocks, 16, 2, 16)
        out_dispatched = impl.forward(
            q,
            k,
            v,
            kv_cache=(k_cache, v_cache),
            attn_metadata=DummyMetadata(),
        )
        assert out_dispatched.shape == (B, T, D)

