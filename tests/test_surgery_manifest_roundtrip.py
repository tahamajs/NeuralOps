#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

import pytest
import shutil
import tempfile
from pathlib import Path
import torch
import torch.nn as nn

import neuralops
from neuralops.api import transplant_attention
from neuralops.export.exporter import export_model, load_exported_model
from neuralops.export.surgery_manifest import (
    SurgeryManifest,
    build_manifest_from_model,
    record_compression_surgery,
)
from neuralops.compression_engine.svd.aa_svd_compressor import SVDLinear


class ToyAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attention_mask=None, **kwargs):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_model**0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return self.o_proj(out)


class ToyTransformer(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.layer0_attn = ToyAttentionBlock(d_model)
        self.layer0_ffn1 = nn.Linear(d_model, d_model * 2)
        self.layer0_ffn2 = nn.Linear(d_model * 2, d_model)
        self.layer1_attn = ToyAttentionBlock(d_model)
        self.layer1_ffn1 = nn.Linear(d_model, d_model * 2)
        self.layer1_ffn2 = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor):
        x = x + self.layer0_attn(x)
        x = x + self.layer0_ffn2(torch.relu(self.layer0_ffn1(x)))
        x = x + self.layer1_attn(x)
        x = x + self.layer1_ffn2(torch.relu(self.layer1_ffn1(x)))
        return x


def test_surgery_manifest_roundtrip_zero_error():
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = ToyTransformer(d_model=64).to(device)

    model = transplant_attention(
        model,
        target_attention="cosformer",
        d_model=64,
        num_heads=4,
    )

    svd_layer0 = SVDLinear(64, 128, rank=16, bias=True)
    svd_layer1 = SVDLinear(64, 128, rank=32, bias=True)

    model.layer0_ffn1 = svd_layer0
    model.layer1_ffn1 = svd_layer1

    record_compression_surgery(
        model,
        layer_path="layer0_ffn1",
        compressor_name="adaptive_svd",
        in_features=64,
        out_features=128,
        rank=16,
    )
    record_compression_surgery(
        model,
        layer_path="layer1_ffn1",
        compressor_name="adaptive_svd",
        in_features=64,
        out_features=128,
        rank=32,
    )

    model.eval()
    dummy_x = torch.randn(2, 8, 64)
    with torch.no_grad():
        orig_out = model(dummy_x)

    tmp_dir = tempfile.mkdtemp(prefix="neuralops_surgery_test_")
    try:
        export_model(model, output_dir=tmp_dir, format="safetensors")

        manifest_path = Path(tmp_dir) / "neuralops_surgery_config.json"
        assert manifest_path.exists(), "neuralops_surgery_config.json must be generated"

        fresh_base = ToyTransformer(d_model=64).to(device)
        reloaded_model, meta = load_exported_model(tmp_dir, base_model=fresh_base)
        reloaded_model.eval()

        with torch.no_grad():
            reloaded_out = reloaded_model(dummy_x)

        max_diff = (orig_out - reloaded_out).abs().max().item()
        assert max_diff < 1e-6, f"Reloaded model outputs diverged! Max diff: {max_diff}"

        assert reloaded_model.layer0_ffn1.rank == 16
        assert reloaded_model.layer1_ffn1.rank == 32
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
