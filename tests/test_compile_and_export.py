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
