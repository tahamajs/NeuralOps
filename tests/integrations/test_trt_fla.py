#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for TensorRT-LLM plugin hooks and Flash-Linear-Attention (FLA) state conversion.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple
import unittest.mock as mock
import pytest
import torch
import torch.nn as nn

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.integrations.tensorrt_llm import (
    TRTPluginSpec,
    TensorRTLLMPluginBuilder,
    TensorRTRuntimeRunner,
    HAS_TRT,
    HAS_TRT_LLM,
)
from neuralops.integrations.tensorrt_llm_plugin import (
    neuralops_attention,
    python_plugin,
)
from neuralops.integrations.fla.adapter import (
    FLALayerAdapter,
    SUPPORTED_MODEL_FAMILIES,
    CAUSAL_FAMILIES,
    _detect_family,
)
from neuralops.integrations.fla.cache_translator import (
    _translate_fla_state_to_hf_cache,
    _translate_hf_cache_to_fla_state,
)


# ==============================================================================
# TensorRT-LLM Tests
# ==============================================================================


class TestTensorRTLLMPlugin:
    """Test suite for TensorRT-LLM plugin specs, code generation, and fail-closed contracts."""

    def test_plugin_spec_instantiation(self) -> None:
        """TRTPluginSpec correctly initializes and stores metadata."""
        spec = TRTPluginSpec(
            name="CustomPlugin",
            plugin_type="NeuralOpsCustom",
            version="2",
            namespace="neuralops_test",
            fields={"dim": 64, "factor": 0.5, "tag": "test"},
        )
        assert spec.name == "CustomPlugin"
        assert spec.plugin_type == "NeuralOpsCustom"
        assert spec.version == "2"
        assert spec.namespace == "neuralops_test"
        assert spec.fields["dim"] == 64
        assert spec.fields["factor"] == 0.5

    def test_plugin_spec_fallback_when_no_trt(self) -> None:
        """TRTPluginSpec gracefully returns None when TensorRT is not available."""
        with mock.patch("neuralops.integrations.tensorrt_llm.HAS_TRT", False):
            spec = TRTPluginSpec("Plugin", "Type")
            assert spec.to_trt_plugin_field_collection() is None
            assert spec.create_trt_plugin() is None

    def test_plugin_builder_initialization(self) -> None:
        """TensorRTLLMPluginBuilder registers default plugin specifications."""
        builder = TensorRTLLMPluginBuilder()
        assert "svd_linear_attention" in builder.plugins
        assert "low_rank_gemm" in builder.plugins
        assert "quantized_w4a16" in builder.plugins

        svd_spec = builder.plugins["svd_linear_attention"]
        assert svd_spec.name == "SVDLinearAttentionPlugin"
        assert svd_spec.plugin_type == "NeuralOpsSVDLinearAttention"

    def test_plugin_builder_availability_check(self) -> None:
        """Check static availability flags."""
        assert TensorRTLLMPluginBuilder.is_available() == HAS_TRT
        assert TensorRTLLMPluginBuilder.is_trt_llm_available() == HAS_TRT_LLM

    def test_generate_cpp_plugin_source(self, tmp_path) -> None:
        """TensorRTLLMPluginBuilder successfully emits valid C++/CUDA source files and CMakeLists."""
        builder = TensorRTLLMPluginBuilder(plugin_dir=str(tmp_path))
        files = builder.generate_cpp_plugin_source(tmp_path)

        assert "svd_linear_attention_plugin.h" in files
        assert "svd_linear_attention_plugin.cu" in files
        assert "CMakeLists.txt" in files

        header_path = tmp_path / "svd_linear_attention_plugin.h"
        cu_path = tmp_path / "svd_linear_attention_plugin.cu"
        cmake_path = tmp_path / "CMakeLists.txt"

        assert header_path.is_file()
        assert cu_path.is_file()
        assert cmake_path.is_file()

        header_content = header_path.read_text(encoding="utf-8")
        assert "class SVDLinearAttentionPlugin : public nvinfer1::IPluginV2DynamicExt" in header_content

        cu_content = cu_path.read_text(encoding="utf-8")
        assert "REGISTER_TENSORRT_PLUGIN(SVDLinearAttentionPluginCreator)" in cu_content

        cmake_content = cmake_path.read_text(encoding="utf-8")
        assert "project(neuralops_tensorrt_plugins LANGUAGES CXX CUDA)" in cmake_content

    def test_fail_closed_stubs_in_builder(self, tmp_path) -> None:
        """Unimplemented engine compilation methods raise fail-closed IntegrationNotImplementedError."""
        builder = TensorRTLLMPluginBuilder()

        with pytest.raises(IntegrationNotImplementedError, match="NeuralOps-to-TensorRT graph translation"):
            builder.build_network_definition(builder=None, network=None)

        with pytest.raises(IntegrationNotImplementedError, match="TensorRT runtime runner is not implemented"):
            builder.get_runtime_runner("dummy.engine")

        with pytest.raises(IntegrationNotImplementedError):
            builder.build_plugin()

        with pytest.raises(IntegrationNotImplementedError):
            builder.export_engine(tmp_path / "dummy.engine")

    def test_tensorrt_runtime_runner_fail_closed(self, tmp_path) -> None:
        """TensorRTRuntimeRunner requires valid engine file and rejects legacy placeholders."""
        non_existent = tmp_path / "non_existent.engine"
        if not HAS_TRT:
            with pytest.raises(ImportError, match="TensorRT is required"):
                TensorRTRuntimeRunner(non_existent)
        else:
            with pytest.raises(FileNotFoundError):
                TensorRTRuntimeRunner(non_existent)

            dummy_legacy = tmp_path / "legacy.engine"
            dummy_legacy.write_bytes(b"NEURALOPS_TRT_ENGINE_V1")
            with pytest.raises(ValueError, match="legacy NeuralOps placeholder"):
                TensorRTRuntimeRunner(dummy_legacy)

    def test_neuralops_attention_plugin_forward(self) -> None:
        """Test TensorRT-LLM python plugin operator forward pass on CPU tensors."""
        b, h, t, d = 2, 4, 16, 32
        q = torch.randn(b, h, t, d)
        k = torch.randn(b, h, t, d)
        v = torch.randn(b, h, t, d)

        # 4D forward
        out_4d = neuralops_attention(q, k, v)
        assert out_4d.shape == (b, h, t, d)
        assert torch.isfinite(out_4d).all()

        # 3D forward (B, T, D)
        q3 = torch.randn(b, t, d)
        k3 = torch.randn(b, t, d)
        v3 = torch.randn(b, t, d)
        out_3d = neuralops_attention(q3, k3, v3)
        assert out_3d.shape == (b, t, d)
        assert torch.isfinite(out_3d).all()

        # Masked forward
        mask = torch.zeros((t, t))
        out_masked = neuralops_attention(q, k, v, mask=mask)
        assert out_masked.shape == (b, h, t, d)
        assert torch.isfinite(out_masked).all()


# ==============================================================================
# Flash-Linear-Attention (FLA) Tests
# ==============================================================================


class TestFLAStateConversionAndAdapter:
    """Test suite for FLA state conversion and FLALayerAdapter contracts."""

    def test_detect_family(self) -> None:
        """Model family detection from config or explicit argument."""
        assert _detect_family(None, parent_family="LLaMA") == "llama"
        assert _detect_family(None, parent_family="Mistral") == "mistral"
        assert _detect_family(None, parent_family="RoBERTa") == "roberta"

        class DummyConfig:
            model_type = "llama"

        assert _detect_family(DummyConfig()) == "llama"

        class DummyQwenConfig:
            model_type = "qwen2"

        assert _detect_family(DummyQwenConfig()) == "qwen2"

        class UnknownConfig:
            model_type = "unknown_net"

        assert _detect_family(UnknownConfig()) == "unknown"

    def test_translate_hf_cache_to_fla_state_none(self) -> None:
        """None past_key_value maps to None recurrent state."""
        assert _translate_hf_cache_to_fla_state(None) is None

    def test_translate_hf_cache_to_fla_state_tensor(self) -> None:
        """Associative state tensor of 3D or 4D passes through unchanged."""
        state_4d = torch.randn(2, 4, 32, 32)
        translated = _translate_hf_cache_to_fla_state(state_4d)
        assert translated is state_4d

        state_3d = torch.randn(2, 32, 32)
        translated_3d = _translate_hf_cache_to_fla_state(state_3d)
        assert translated_3d is state_3d

        # 2D tensor is invalid as associative recurrent state
        with pytest.raises(IntegrationNotImplementedError, match="Unsupported FLA recurrent state tensor dimensions"):
            _translate_hf_cache_to_fla_state(torch.randn(2, 32))

    def test_translate_hf_cache_to_fla_state_empty_kv(self) -> None:
        """Empty sequence past_key_values tuple returns None."""
        empty_k = torch.empty((2, 4, 0, 32))
        empty_v = torch.empty((2, 4, 0, 32))
        assert _translate_hf_cache_to_fla_state((empty_k, empty_v)) is None

    def test_translate_hf_cache_to_fla_state_non_empty_kv_raises(self) -> None:
        """Non-empty standard softmax KV cache raises fail-closed exception."""
        k = torch.randn(2, 4, 16, 32)
        v = torch.randn(2, 4, 16, 32)
        with pytest.raises(IntegrationNotImplementedError, match="Converting historical 4D"):
            _translate_hf_cache_to_fla_state((k, v))

    def test_translate_hf_cache_unsupported_object_raises(self) -> None:
        """Arbitrary objects fail closed with informative error."""

        class DynamicCache:
            pass

        with pytest.raises(IntegrationNotImplementedError, match="Unsupported Hugging Face cache type 'DynamicCache'"):
            _translate_hf_cache_to_fla_state(DynamicCache())

    def test_translate_fla_state_to_hf_cache(self) -> None:
        """Next state translation preserves tensors and tuples."""
        assert _translate_fla_state_to_hf_cache(None) is None
        state = torch.randn(2, 4, 32, 32)
        assert _translate_fla_state_to_hf_cache(state) is state
        state_tuple = (state, state)
        assert _translate_fla_state_to_hf_cache(state_tuple) == state_tuple

    def test_fla_layer_adapter_fail_closed_without_fla(self) -> None:
        """Adapter fails closed if fla is not available."""
        with mock.patch("neuralops.integrations.fla.fla_is_available", return_value=False):
            with pytest.raises(IntegrationNotImplementedError, match="The 'fla' package is required"):
                FLALayerAdapter(fla_layer_cls=None, parent_family="llama")

    def test_fla_layer_adapter_unsupported_family_raises(self) -> None:
        """Adapter rejects unsupported model families."""
        with mock.patch("neuralops.integrations.fla.fla_is_available", return_value=True):
            with pytest.raises(
                IntegrationNotImplementedError, match="Unsupported parent model family 'unsupported_arch'"
            ):
                FLALayerAdapter(fla_layer_cls=None, parent_family="unsupported_arch")

    def test_fla_layer_adapter_execution_causal_family(self) -> None:
        """Adapter correctly handles causal family arity (3) and wraps inner layer."""

        class MockFLALayer(nn.Module):
            def __init__(self, hidden_size: int = 64, num_heads: int = 4, **kwargs: Any) -> None:
                super().__init__()
                self.proj = nn.Linear(hidden_size, hidden_size)

            def forward(
                self,
                hidden_states: torch.Tensor,
                past_key_values: Optional[torch.Tensor] = None,
                **kwargs: Any,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                out = self.proj(hidden_states)
                next_state = (
                    past_key_values + 1.0
                    if past_key_values is not None
                    else torch.zeros((hidden_states.size(0), 4, 16, 16))
                )
                return out, next_state

        with mock.patch("neuralops.integrations.fla.fla_is_available", return_value=True):
            adapter = FLALayerAdapter(
                fla_layer_cls=MockFLALayer,
                parent_family="llama",
                hidden_size=64,
                num_heads=4,
            )
            assert adapter.return_arity == 3
            assert adapter.family == "llama"

            inputs = torch.randn(2, 8, 64)
            # 1. Forward with no cache
            out, attn_weights, next_cache = adapter(inputs)
            assert out.shape == (2, 8, 64)
            assert attn_weights is None
            assert next_cache is not None
            assert next_cache.shape == (2, 4, 16, 16)

            # 2. Forward with existing cache
            out2, _, next_cache2 = adapter(inputs, past_key_value=next_cache)
            assert out2.shape == (2, 8, 64)
            assert torch.allclose(next_cache2, next_cache + 1.0)

    def test_fla_layer_adapter_execution_encoder_family(self) -> None:
        """Adapter correctly handles encoder family arity (2) (e.g. BERT/RoBERTa)."""

        class MockFLAEncoderLayer(nn.Module):
            def __init__(self, hidden_size: int = 64, num_heads: int = 4, **kwargs: Any) -> None:
                super().__init__()
                self.proj = nn.Linear(hidden_size, hidden_size)

            def forward(
                self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs: Any,
            ) -> torch.Tensor:
                out = self.proj(hidden_states)
                return out

        with mock.patch("neuralops.integrations.fla.fla_is_available", return_value=True):
            adapter = FLALayerAdapter(
                fla_layer_cls=MockFLAEncoderLayer,
                parent_family="bert",
                hidden_size=64,
                num_heads=4,
            )
            assert adapter.return_arity == 2
            assert adapter.family == "bert"

            inputs = torch.randn(2, 8, 64)
            res = adapter(inputs)
            assert len(res) == 2
            out, probs = res
            assert out.shape == (2, 8, 64)
            assert probs is None
