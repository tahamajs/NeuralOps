#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for neuralops.core.extensibility contract and dynamic validators."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.core.extensibility import (
    get_custom_metadata,
    register_custom_attention,
    register_custom_compressor,
    scaffold_module,
    unregister_custom_module,
    validate_module_contract,
)
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY


def test_validate_module_contract_builtin() -> None:
    """Verify 6-point contract validator on canonical built-in module."""
    results = validate_module_contract("full_attention", embed_dim=64, num_heads=4, seq_len=16, batch_size=2)
    assert results["total_checks"] == 6
    assert results["checks"]["1_instantiation"]["status"] == "PASSED"
    assert results["checks"]["2_forward_shape"]["status"] == "PASSED"
    assert results["checks"]["3_backward_gradients"]["status"] == "PASSED"
    assert results["checks"]["4_introspection_metadata"]["status"] == "PASSED"
    assert results["checks"]["5_surgical_micro_healing"]["status"] == "PASSED"
    assert results["checks"]["6_persistence_roundtrip"]["status"] == "PASSED"
    assert results["all_passed"] is True


def test_validate_module_contract_custom_registered() -> None:
    """Verify 6-point contract validator on custom registered module."""
    key = "test_custom_contract_attn"

    @register_custom_attention(key, family="linear-scan")
    class CustomContractAttn(nn.Module):
        def __init__(self, embed_dim: int = 64, num_heads: int = 4):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            return self.out_proj(q + k + v)

    try:
        results = validate_module_contract(key, embed_dim=64, num_heads=4, seq_len=16, batch_size=2)
        assert results["total_checks"] == 6
        assert results["all_passed"] is True
        assert results["checks"]["4_introspection_metadata"]["family"] == "linear-scan"
    finally:
        unregister_custom_module(key, category="attention")


def test_validate_module_contract_unregistered_module_instance() -> None:
    """Verify 6-point contract validator on unregistered nn.Module instance with default spec."""

    class CustomUnregistered(nn.Module):
        def __init__(self, embed_dim: int = 64, num_heads: int = 4):
            super().__init__()
            self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return self.proj(x)

    mod = CustomUnregistered(64, 4)
    results = validate_module_contract(mod, embed_dim=64, num_heads=4, seq_len=16, batch_size=2)
    assert results["total_checks"] == 6
    assert results["checks"]["1_instantiation"]["status"] == "PASSED"
    assert results["checks"]["2_forward_shape"]["status"] == "PASSED"
    assert results["checks"]["3_backward_gradients"]["status"] == "PASSED"
    assert results["checks"]["4_introspection_metadata"]["status"] == "PASSED"
    assert results["checks"]["5_surgical_micro_healing"]["status"] == "PASSED"
    # An unregistered module cannot be restored from a named registry checkpoint without registration
    assert results["checks"]["6_persistence_roundtrip"]["status"] == "FAILED"


def test_scaffold_module_attention_and_compressor() -> None:
    """Verify template generator generates valid code."""
    attn_code = scaffold_module("test_generated_attn", category="attention", family="linear-scan")
    assert "class TestGeneratedAttn(nn.Module):" in attn_code
    assert "@neuralops.register_custom_attention" in attn_code

    comp_code = scaffold_module("test_generated_comp", category="compressor", family="low-rank")
    assert "class TestGeneratedComp:" in comp_code
    assert "@neuralops.register_custom_compressor" in comp_code


def test_custom_registration_lifecycle() -> None:
    """Verify registering, retrieving metadata, and unregistering cleanly."""
    attn_key = "temp_custom_lifecycle_attn"

    @register_custom_attention(attn_key, family="state-space")
    class LifecycleAttn(nn.Module):
        pass

    assert attn_key in ATTENTION_REGISTRY
    meta = get_custom_metadata(attn_key)
    assert meta["family"] == "state-space"

    unreg = unregister_custom_module(attn_key, category="attention")
    assert unreg is True
    assert attn_key not in ATTENTION_REGISTRY
    assert get_custom_metadata(attn_key) == {}
