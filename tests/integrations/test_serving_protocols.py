#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and protocol compliance tests for serving integrations and runtime capability probing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import pytest
import torch
import torch.nn as nn

from neuralops.inference.engine import ContinuousBatcher
from neuralops.inference.paged_cache import PagedKVCache, PagedKVCacheManager
from neuralops.integrations.capabilities import (
    BackendCapabilities,
    probe_serving_ecosystem,
    query_serving_capabilities,
)
from neuralops.integrations.protocols import (
    ContinuousBatcherProtocol,
    PagedKVCacheProtocol,
    RadixTrieProtocol,
    ServingEngineProtocol,
)
from neuralops.integrations.radix_prefix_cache import SGLangRadixAdapter


class DummyServingEngine:
    """Mock implementation fulfilling ServingEngineProtocol."""

    def __init__(self) -> None:
        self.initialized = False
        self.active_kv: Optional[torch.Tensor] = None

    def initialize(self, *args: Any, **kwargs: Any) -> Any:
        self.initialized = True
        return True

    def prefill(
        self,
        input_ids: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not self.initialized:
            raise RuntimeError("Engine not initialized.")
        self.active_kv = torch.zeros(input_ids.shape[0], 1, 16)
        return self.active_kv

    def decode_step(self, *args: Any, **kwargs: Any) -> Any:
        return torch.tensor([[1]])

    def shutdown(self) -> None:
        self.initialized = False
        self.active_kv = None


class IncompleteEngine:
    """Negative test class missing shutdown."""

    def initialize(self) -> None:
        pass

    def prefill(self, input_ids: torch.Tensor) -> Any:
        pass

    def decode_step(self) -> Any:
        pass


def test_serving_engine_protocol_runtime_checkable():
    engine = DummyServingEngine()
    assert isinstance(engine, ServingEngineProtocol)

    incomplete = IncompleteEngine()
    assert not isinstance(incomplete, ServingEngineProtocol)


def test_paged_kv_cache_protocol_runtime_checkable():
    cache = PagedKVCache(
        num_blocks=8,
        block_size=16,
        num_heads=2,
        head_dim=16,
        dtype=torch.float32,
        device="cpu",
    )
    assert isinstance(cache, PagedKVCacheProtocol)

    # Test alias compatibility
    manager = PagedKVCacheManager(
        num_blocks=8,
        block_size=16,
        num_heads=2,
        head_dim=16,
        dtype=torch.float32,
        device="cpu",
    )
    assert isinstance(manager, PagedKVCacheProtocol)


def test_continuous_batcher_protocol_runtime_checkable():
    dummy_model = nn.Linear(16, 16)
    batcher = ContinuousBatcher(model=dummy_model, max_batch_size=4)
    assert isinstance(batcher, ContinuousBatcherProtocol)


def test_radix_trie_protocol_runtime_checkable():
    radix_adapter = SGLangRadixAdapter(max_cached_prefixes=64, device="cpu")
    assert isinstance(radix_adapter, RadixTrieProtocol)


def test_probe_serving_ecosystem_structure():
    ecosystem = probe_serving_ecosystem()
    assert isinstance(ecosystem, dict)

    # Required keys
    assert "vllm" in ecosystem
    assert "sglang" in ecosystem
    assert "tgi" in ecosystem
    assert "triton" in ecosystem

    # Check vllm entry
    vllm_info = ecosystem["vllm"]
    assert "is_available" in vllm_info
    assert "version" in vllm_info
    assert "has_cuda_ops" in vllm_info
    assert isinstance(vllm_info["is_available"], bool)
    assert isinstance(vllm_info["has_cuda_ops"], bool)

    # Check sglang entry
    sglang_info = ecosystem["sglang"]
    assert "is_available" in sglang_info
    assert "version" in sglang_info
    assert "supports_radix" in sglang_info
    assert isinstance(sglang_info["is_available"], bool)
    assert isinstance(sglang_info["supports_radix"], bool)

    # Check tgi entry
    tgi_info = ecosystem["tgi"]
    assert "is_available" in tgi_info
    assert "version" in tgi_info
    assert "has_grpc" in tgi_info
    assert isinstance(tgi_info["is_available"], bool)
    assert isinstance(tgi_info["has_grpc"], bool)

    # Check triton entry
    triton_info = ecosystem["triton"]
    assert "is_available" in triton_info
    assert "version" in triton_info
    assert "jit_capable" in triton_info
    assert "target_compute_capability" in triton_info
    assert isinstance(triton_info["is_available"], bool)
    assert isinstance(triton_info["jit_capable"], bool)


def test_query_serving_capabilities_matches_ecosystem():
    caps = query_serving_capabilities()
    assert isinstance(caps, dict)
    assert "vllm" in caps
    assert "sglang" in caps
    assert "tensorrt_llm" in caps
    assert "tgi" in caps
    assert "neuralops_native" in caps

    for name, cap in caps.items():
        assert isinstance(cap, BackendCapabilities)
        assert cap.name == name
        assert isinstance(cap.is_available, bool)
        assert isinstance(cap.supports_paged_attention, bool)
        assert isinstance(cap.supports_speculative_decoding, bool)
        assert isinstance(cap.install_hint, str)


def test_probe_serving_ecosystem_mocking(monkeypatch: pytest.MonkeyPatch):
    # Simulate an environment where vllm and triton are present with cuda
    def mock_find_spec(module_name: str) -> bool:
        return module_name in (
            "vllm",
            "vllm._C",
            "triton",
            "text_generation_server",
            "sglang",
            "sglang.srt",
        )

    monkeypatch.setattr("neuralops.integrations.capabilities._safe_find_spec", mock_find_spec)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (8, 9))

    ecosystem = probe_serving_ecosystem()
    assert ecosystem["vllm"]["is_available"] is True
    assert ecosystem["vllm"]["has_cuda_ops"] is True
    assert ecosystem["sglang"]["is_available"] is True
    assert ecosystem["sglang"]["supports_radix"] is True
    assert ecosystem["triton"]["is_available"] is True
    assert ecosystem["triton"]["target_compute_capability"] == (8, 9)
