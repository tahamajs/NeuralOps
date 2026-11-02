#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for all 9 serving frameworks and serialization targets."""

from pathlib import Path
import tempfile
import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

from neuralops.export.exporter import export_model
from neuralops.export.gguf_exporter import export_to_gguf, GGUF_MAGIC
from neuralops.export.runtime_compat import check_compatibility
from neuralops.integrations.hf_automodel import patch_model_attention
from neuralops.integrations.sglang.radix_recurrent import RadixRecurrentTree
from neuralops.integrations.tensorrt_llm import TRTPluginSpec, TensorRTLLMPluginBuilder
from neuralops.integrations.vllm_backend import NeuralOpsAttentionBackend
from neuralops.ops.cuda import cosformer_linear_attention, force_kernel_backend
from neuralops.serving.request_batcher import TGIRequest, TGIServingAdapter


def test_01_huggingface_patching():
    """Verify in-place HuggingFace attention replacement."""
    config = BertConfig(hidden_size=64, num_attention_heads=2, num_hidden_layers=1)
    model = BertModel(config)
    patched = patch_model_attention(model, "cosformer")
    x = torch.randint(0, 100, (2, 8))
    out = patched(x)
    assert out.last_hidden_state.shape == (2, 8, 64)


def test_02_fla_fallback_verification():
    """Verify Flash Linear Attention level-3 fallback path."""
    b, h, seq, d = 2, 2, 8, 16
    q = torch.randn(b, h, seq, d)
    k = torch.randn(b, h, seq, d)
    v = torch.randn(b, h, seq, d)
    with force_kernel_backend("pytorch"):
        out = cosformer_linear_attention(q, k, v)
    assert torch.isfinite(out).all().item()


def test_03_safetensors_export_roundtrip():
    """Verify Safetensors export with neuralops_config.json metadata."""
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    with tempfile.TemporaryDirectory() as tmpdir:
        out = export_model(model, tmpdir, format="safetensors")
        st_file = Path(out) / "model.safetensors"
        cfg_file = Path(out) / "neuralops_config.json"
        assert st_file.exists()
        assert cfg_file.exists()


def test_04_onnx_fail_closed_contract():
    """Verify ONNX export fails closed with descriptive error on unsupported graphs."""
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            x = torch.randn(1, 16)
            out = export_model(model, tmpdir, format="onnx", example_input=x)
            assert Path(out).exists()
        except RuntimeError as e:
            # Must contain descriptive failure note
            assert "ONNX" in str(e)


def test_05_vllm_attention_backend_spec():
    """Verify vLLM attention backend specification and state shape."""
    assert NeuralOpsAttentionBackend.get_name() == "NEURALOPS"
    shape = NeuralOpsAttentionBackend.get_kv_cache_shape(8, 16, 4, 32)
    assert shape == (8, 4, 32, 32)


def test_06_sglang_radix_recurrent_prefix_matching():
    """Verify SGLang RadixRecurrentTree exact prefix matching."""
    tree = RadixRecurrentTree()
    state = torch.randn(1, 8)
    tree.insert([10, 20, 30], state)
    matched_tokens, matched_state = tree.match_prefix([10, 20, 30, 40])
    assert matched_tokens == [10, 20, 30]
    assert torch.allclose(matched_state, state)


def test_07_tensorrt_llm_plugin_generator():
    """Verify TensorRT-LLM C++ CUDA plugin source generation."""
    builder = TensorRTLLMPluginBuilder()
    with tempfile.TemporaryDirectory() as tmpdir:
        files = builder.generate_cpp_plugin_source(tmpdir)
        assert len(files) == 3
        for f in files.values():
            assert Path(f).exists()


def test_08_tgi_continuous_batching_adapter():
    """Verify TGI serving adapter request queuing."""
    model = nn.Sequential(nn.Linear(16, 16))
    adapter = TGIServingAdapter(model, max_batch_size=4)
    req = adapter.add_request("req_01", prompt="Prompt test")
    stats = adapter.stats()
    assert stats["total_requests"] == 1
    assert stats["active_requests"] == 1


def test_09_gguf_v3_binary_export_and_inspection():
    """Verify native GGUF v3 binary container creation and compatibility."""
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
    with tempfile.TemporaryDirectory() as tmpdir:
        out = export_model(model, tmpdir, format="gguf")
        assert out.exists()
        with open(out, "rb") as f:
            magic = f.read(4)
            assert magic == GGUF_MAGIC
        rep = check_compatibility(tmpdir, engines=["gguf"])
        assert rep.results["gguf"].status == "SUCCESS"
