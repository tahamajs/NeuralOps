#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Comprehensive test suite for NeuralOps v1.7.0 core hardening and ergonomics.

Tests:
1. Fast-path cached call mode in `UniversalAttentionWrapper`:
   - Verifying `_cached_dispatch_mode` is None prior to execution.
   - Verifying `_cached_dispatch_mode` is populated on the first call and reused on subsequent calls without throwing.
   - Verifying correct dispatch across varying attention call signatures.
   - Verifying cache invalidation on signature mismatch (TypeError recovery).
2. Multi-batch calibration iteration in `_heal_replaced_model_impl`:
   - Multi-batch list of tensor calibration inputs.
   - Multi-batch list of dictionary calibration inputs.
   - Student parameter trainability preservation through `heal_replaced_model`.
3. Structured fallback logging in `neuralops.ops.triton.low_rank`:
   - Dynamic FP8 quantization Triton kernel failure fallback logging.
   - Dynamic FP8 quantization float8 cast failure fallback logging.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from neuralops.core._attention_surgery import _heal_replaced_model_impl, heal_replaced_model
from neuralops.ops.triton.low_rank import quantize_to_fp8_dynamic
from neuralops.utils.replace_attention import UniversalAttentionWrapper


# ---------------------------------------------------------------------------
# Test Helpers & Dummy Modules
# ---------------------------------------------------------------------------


class DummyStandardAttention(nn.Module):
    """Attention layer supporting standard kwargs."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        out = self.proj(hidden_states)
        if attention_mask is not None and attention_mask.shape == out.shape:
            out = out + attention_mask
        return out


class DummyStrictAttention(nn.Module):
    """Attention layer with a strict single positional argument signature."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class DummyQKVAttention(nn.Module):
    """Attention layer expecting query, key, value signature."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.proj(query)


class DummyToyAttentionLayer(nn.Module):
    """Toy transformer layer exposing a named self_attn module for healing discovery."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.self_attn = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn(x)


class DummyToyTransformer(nn.Module):
    """Toy multi-layer transformer for testing healing procedures."""

    def __init__(self, num_layers: int = 2, dim: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DummyToyAttentionLayer(dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class DummyToyDictTransformer(nn.Module):
    """Toy transformer accepting dictionary kwargs."""

    def __init__(self, num_layers: int = 2, dim: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DummyToyAttentionLayer(dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# 1. Fast-Path Cached Call Mode in UniversalAttentionWrapper
# ---------------------------------------------------------------------------


def test_universal_attention_wrapper_cached_dispatch_mode_standard() -> None:
    """Verify that _cached_dispatch_mode is populated on first call and reused on subsequent calls."""
    wrapped_module = DummyStandardAttention(dim=16)
    wrapper = UniversalAttentionWrapper(wrapped_module)

    # Initial state: no cached mode
    assert getattr(wrapper, "_cached_dispatch_mode", None) is None

    x = torch.randn(2, 4, 16)
    mask = torch.zeros(2, 1, 1, 4)

    # First call populates cache
    out1 = wrapper(x, attention_mask=mask)
    cached_mode = getattr(wrapper, "_cached_dispatch_mode", None)
    assert cached_mode is not None
    assert isinstance(cached_mode, int)

    # Subsequent calls must reuse cache without throwing and yield identical results
    out2 = wrapper(x, attention_mask=mask)
    out3 = wrapper(x, attention_mask=mask)

    assert getattr(wrapper, "_cached_dispatch_mode", None) == cached_mode
    assert torch.allclose(out1[0], out2[0])
    assert torch.allclose(out2[0], out3[0])


def test_universal_attention_wrapper_cached_dispatch_mode_strict() -> None:
    """Verify that a strict forward(hidden_states) signature correctly caches and executes fast-path."""
    wrapped_module = DummyStrictAttention(dim=16)
    wrapper = UniversalAttentionWrapper(wrapped_module)

    assert getattr(wrapper, "_cached_dispatch_mode", None) is None

    x = torch.randn(2, 4, 16)
    out1 = wrapper(x)
    cached_mode = getattr(wrapper, "_cached_dispatch_mode", None)

    assert cached_mode in (3, 7)

    out2 = wrapper(x)
    assert getattr(wrapper, "_cached_dispatch_mode", None) == cached_mode
    assert torch.allclose(out1[0], out2[0])


def test_universal_attention_wrapper_cached_dispatch_mode_qkv() -> None:
    """Verify that query/key/value attention signature populates cached mode 4 and reuses it."""
    wrapped_module = DummyQKVAttention(dim=16)
    wrapper = UniversalAttentionWrapper(wrapped_module)

    assert getattr(wrapper, "_cached_dispatch_mode", None) is None

    x = torch.randn(2, 4, 16)
    out1 = wrapper(x)
    cached_mode = getattr(wrapper, "_cached_dispatch_mode", None)

    assert cached_mode == 4

    out2 = wrapper(x)
    assert getattr(wrapper, "_cached_dispatch_mode", None) == 4
    assert torch.allclose(out1[0], out2[0])


def test_universal_attention_wrapper_cache_invalidation_on_type_error() -> None:
    """Verify that if a cached mode encounters TypeError, cache is invalidated and rediscovers."""
    wrapped_module = DummyStrictAttention(dim=16)
    wrapper = UniversalAttentionWrapper(wrapped_module)

    x = torch.randn(2, 4, 16)
    wrapper(x)
    assert getattr(wrapper, "_cached_dispatch_mode", None) is not None

    # Deliberately corrupt the cached mode to mode 0 (which passes attention_mask and kwargs)
    # Since DummyStrictAttention rejects attention_mask, mode 0 raises TypeError.
    wrapper._cached_dispatch_mode = 0

    # The call should catch TypeError, reset _cached_dispatch_mode to None, and redispatch cleanly
    out = wrapper(x)
    assert out is not None
    assert getattr(wrapper, "_cached_dispatch_mode", None) in (3, 7)


# ---------------------------------------------------------------------------
# 2. Multi-Batch Calibration Iteration in _heal_replaced_model_impl
# ---------------------------------------------------------------------------


def test_heal_replaced_model_multi_batch_tensor_sequence() -> None:
    """Verify that _heal_replaced_model_impl iterates over multi-batch tensor calibration inputs."""
    teacher = DummyToyTransformer(num_layers=2, dim=16)
    student = DummyToyTransformer(num_layers=2, dim=16)

    # Perturb student parameters to simulate post-surgery divergence
    with torch.no_grad():
        for p in student.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    batches = [
        torch.randn(2, 4, 16),
        torch.randn(3, 4, 16),
        torch.randn(2, 4, 16),
    ]

    stats = _heal_replaced_model_impl(
        teacher_model=teacher,
        student_model=student,
        calibration_inputs=batches,
        steps=10,
        lr=1e-2,
        also_ridge_init=False,
        log_to_ledger=False,
    )

    assert stats["num_healed_layers"] == 2
    assert stats["steps"] == 10
    assert "initial_mse" in stats
    assert "final_mse" in stats
    assert stats["final_mse"] <= stats["initial_mse"]
    assert len(stats["per_layer"]) == 2
    for layer_stats in stats["per_layer"].values():
        assert layer_stats["final_mse"] <= layer_stats["initial_mse"]


def test_heal_replaced_model_multi_batch_dict_sequence() -> None:
    """Verify that _heal_replaced_model_impl handles a list of dictionary batches."""
    teacher = DummyToyDictTransformer(num_layers=2, dim=16)
    student = DummyToyDictTransformer(num_layers=2, dim=16)

    batches = [
        {"x": torch.randn(2, 4, 16)},
        {"x": torch.randn(3, 4, 16)},
    ]

    stats = _heal_replaced_model_impl(
        teacher_model=teacher,
        student_model=student,
        calibration_inputs=batches,
        steps=5,
        lr=1e-2,
        also_ridge_init=False,
        log_to_ledger=False,
    )

    assert stats["num_healed_layers"] == 2
    assert stats["final_mse"] >= 0.0


def test_heal_replaced_model_preserves_grad_state() -> None:
    """Verify that heal_replaced_model preserves parameter requires_grad configuration."""
    teacher = DummyToyTransformer(num_layers=1, dim=16)
    student = DummyToyTransformer(num_layers=1, dim=16)

    student.layers[0].self_attn.weight.requires_grad = True
    student.layers[0].self_attn.bias.requires_grad = False

    batches = [torch.randn(2, 4, 16), torch.randn(2, 4, 16)]

    heal_replaced_model(
        teacher_model=teacher,
        student_model=student,
        calibration_inputs=batches,
        steps=5,
        also_ridge_init=False,
        log_to_ledger=False,
    )

    assert student.layers[0].self_attn.weight.requires_grad is True
    assert student.layers[0].self_attn.bias.requires_grad is False


# ---------------------------------------------------------------------------
# 3. Logging of Fallbacks in low_rank.py
# ---------------------------------------------------------------------------


def test_low_rank_dynamic_fp8_kernel_fallback_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Verify that dynamic FP8 Triton kernel failure logs a structured warning and delegates to reference."""
    x = torch.randn(4, 16)

    mock_profile = MagicMock()
    mock_profile.sm_major = 9
    mock_profile.sm_minor = 0

    with (
        patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
        patch(
            "neuralops.ops.triton.hardware_profile.get_device_hardware_profile",
            side_effect=RuntimeError("Simulated Triton hardware profiling failure"),
        ),
        caplog.at_level(logging.WARNING, logger="neuralops.ops.triton.low_rank"),
    ):
        x_fp8, scale = quantize_to_fp8_dynamic(x)

    assert "Dynamic FP8 quantization kernel failed" in caplog.text
    assert "Simulated Triton hardware profiling failure" in caplog.text
    assert "delegating to reference path" in caplog.text
    assert x_fp8.shape == x.shape
    assert scale.shape == (4,)


def test_low_rank_dynamic_fp8_cast_fallback_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Verify that float8_e4m3fn cast failure logs a structured warning and keeps original dtype."""
    x = torch.randn(4, 16, dtype=torch.float32)

    with (
        patch("torch.float8_e4m3fn", "invalid_dtype_that_fails_to"),
        caplog.at_level(logging.WARNING, logger="neuralops.ops.triton.low_rank"),
    ):
        x_fp8, scale = quantize_to_fp8_dynamic(x)

    assert "Cast to float8_e4m3fn failed" in caplog.text
    assert "keeping original dtype" in caplog.text
    assert x_fp8.dtype == torch.float32
    assert x_fp8.shape == x.shape
    assert scale.shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
