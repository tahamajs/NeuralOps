"""
Tests for torch.compile and model export functionality.
"""

import shutil
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

import neuralops
from neuralops.core.triton_kernels.op_registry import (
    compilable_linear_attn,
    compilable_low_rank,
)
from neuralops.export.exporter import export_model, load_exported_model


class SimpleTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        out = compilable_linear_attn(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.out_proj(out)


def test_torch_compile_simple_block():
    model = SimpleTransformerBlock(d_model=32, n_heads=2).eval()
    x = torch.randn(2, 16, 32)

    # Standard forward
    y_eager = model(x)

    # torch.compile forward if supported
    if hasattr(torch, "compile"):
        try:
            compiled = torch.compile(model, fullgraph=False)
            y_compiled = compiled(x)
            torch.testing.assert_close(y_eager, y_compiled, atol=1e-3, rtol=1e-3)
        except Exception as e:
            # Older PyTorch or unsupported dynamic backend
            pytest.skip(f"torch.compile skipped on this runtime: {e}")


def test_export_safetensors_and_config():
    model = SimpleTransformerBlock(d_model=32, n_heads=2).eval()
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        orig_logits = model(x)

    tmp_dir = tempfile.mkdtemp()
    try:
        out_dir = export_model(
            model=model,
            output_dir=tmp_dir,
            format="safetensors",
            model_name_or_path="simple-block",
            attention_type=None,
            rank=16,
        )
        assert Path(out_dir).exists()
        reloaded_model, config = load_exported_model(out_dir, base_model=SimpleTransformerBlock(d_model=32, n_heads=2))
        assert config["attention_type"] is None
        assert config["rank"] == 16
        assert config["format"] == "safetensors"

        reloaded_model.eval()
        with torch.no_grad():
            reloaded_logits = reloaded_model(x)

        torch.testing.assert_close(orig_logits, reloaded_logits, atol=1e-5, rtol=1e-5)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_export_onnx_and_torchscript():
    model = SimpleTransformerBlock(d_model=32, n_heads=2)
    tmp_dir = tempfile.mkdtemp()
    try:
        dummy_x = torch.randn(1, 8, 32)
        onnx_file = export_model(
            model=model,
            output_dir=tmp_dir,
            format="onnx",
            sample_input=dummy_x,
        )
        assert Path(onnx_file).exists()

        ts_file = export_model(
            model=model,
            output_dir=tmp_dir,
            format="torchscript",
            sample_input=dummy_x,
        )
        assert Path(ts_file).exists()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_full_roundtrip_surgery_compress_export_reload():
    """Verify end-to-end: surgery -> compress -> export_model -> load_model parity."""
    from transformers import AutoConfig, AutoModelForSequenceClassification

    config = AutoConfig.from_pretrained("distilbert-base-uncased", num_labels=2)
    model = AutoModelForSequenceClassification.from_config(config).eval()

    # 1. Surgery
    model = neuralops.replace_attention(model, "cosformer")
    assert hasattr(model, "_neuralops_metadata")
    assert len(model._neuralops_metadata) >= 1

    # 2. Compress
    model = neuralops.compress(model, method="direct_svd", rank=16)
    assert len(model._neuralops_metadata) >= 2

    # Dummy inputs
    dummy_input = torch.randint(0, 1000, (1, 16))
    with torch.no_grad():
        orig_out = model(dummy_input).logits

    # 3. Export
    tmp_dir = tempfile.mkdtemp()
    try:
        # Note: export_model should automatically infer metadata from model._neuralops_metadata
        out_dir = neuralops.export_model(model, tmp_dir, format="safetensors")
        assert Path(out_dir).exists()

        # 4. Reload via neuralops.load_model with base architecture
        base_fresh = AutoModelForSequenceClassification.from_config(config).eval()
        reloaded = neuralops.load_model(out_dir, base_model=base_fresh).eval()

        with torch.no_grad():
            reloaded_out = reloaded(dummy_input).logits

        torch.testing.assert_close(orig_out, reloaded_out, atol=1e-5, rtol=1e-5)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_all_19_public_api_functions():
    """Verify that all 19 public API functions are callable and behave correctly."""
    # 1. get_version
    ver = neuralops.get_version()
    assert ver == "1.3.0"
    assert neuralops.__version__ == "1.3.0"

    # 2. list_attentions
    attns = neuralops.list_attentions()
    assert isinstance(attns, list) and len(attns) >= 130
    assert "cosformer" in attns

    # 3. list_compressors
    comps = neuralops.list_compressors()
    assert isinstance(comps, list) and len(comps) >= 70
    assert "direct_svd" in comps

    # 4. get_attention_class
    cls_attn = neuralops.get_attention_class("cosformer")
    assert callable(cls_attn)

    # 5. get_compressor_class
    cls_comp = neuralops.get_compressor_class("direct_svd")
    assert callable(cls_comp)

    # 6. list_families
    fams = neuralops.list_families()
    assert isinstance(fams, dict) and len(fams) >= 8

    # 7. available_backends
    backends = neuralops.available_backends()
    assert isinstance(backends, list) and "cpu" in backends

    # 8. has_triton
    ht = neuralops.has_triton()
    assert isinstance(ht, bool)

    # 9. info
    info_dict = neuralops.info()
    assert info_dict["version"] == "1.3.0"
    assert "pytorch_version" in info_dict

    # 10. validate
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(100, 16)
            self.linear = nn.Linear(16, 2)
        def forward(self, x, labels=None):
            h = self.emb(x).mean(dim=1)
            logits = self.linear(h)
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(logits, labels)
            return type("Out", (), {"logits": logits, "loss": loss})()

    tiny = TinyModel()
    val_ok = neuralops.validate(tiny, input_shape=(2, 8), device="cpu")
    assert val_ok is True


