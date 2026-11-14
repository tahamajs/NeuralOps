#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and regression tests for public API replace_attention, compose_attention, and compress."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.api import (
    compose_attention,
    compress,
    replace_attention,
    transplant,
    transplant_attention,
)


class DummySelfAttention(nn.Module):
    def __init__(self, d_model=32, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, attention_mask=None, **kwargs):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (self.d_model**0.5)
        attn_probs = torch.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_probs, v)
        return self.out_proj(out)


class DummyTransformerModel(nn.Module):
    def __init__(self, num_layers=2, d_model=32):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attention": DummySelfAttention(d_model=d_model),
                        "mlp": nn.Linear(d_model, d_model),
                    }
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            attn = layer["attention"]
            attn_out = attn(x)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            x = x + attn_out
            x = x + layer["mlp"](x)
        return x


class TestReplaceAttentionAPI:
    """Verifies replace_attention and transplant_attention API surface."""

    def test_replace_attention_with_registered_module(self):
        model = DummyTransformerModel(num_layers=2, d_model=32)
        replaced = replace_attention(model, attention="linear_attention")
        assert hasattr(replaced, "_neuralops_metadata")
        assert len(replaced._neuralops_metadata) > 0
        assert replaced._neuralops_metadata[-1]["op"] == "replace"

        # Verify forward execution
        x = torch.randn(2, 8, 32)
        out = replaced(x)
        assert out.shape == (2, 8, 32)

    def test_replace_attention_with_custom_module(self):
        model = DummyTransformerModel(num_layers=2, d_model=32)
        custom_mod = nn.Linear(32, 32)
        replaced = replace_attention(model, target_module=custom_mod)
        assert hasattr(replaced, "_neuralops_metadata")
        assert replaced._neuralops_metadata[-1]["type"] == "Linear"

    def test_transplant_attention_alias(self):
        model = DummyTransformerModel(num_layers=1, d_model=32)
        res = transplant_attention(model, attention="linear_attention")
        assert res is not None
        assert transplant is transplant_attention

    def test_layer_filtering(self):
        model = DummyTransformerModel(num_layers=2, d_model=32)
        replaced = replace_attention(model, attention="linear_attention", layers=[0])
        assert hasattr(replaced, "_neuralops_metadata")
        assert replaced._neuralops_metadata[-1]["kwargs"]["layers"] == [0]

    @pytest.mark.parametrize("attn_name", ["full_attention", "cosformer", "gla"])
    def test_replace_attention_three_types(self, attn_name: str):
        torch.manual_seed(42)
        model = DummyTransformerModel(num_layers=2, d_model=32)
        replaced = replace_attention(model, attention=attn_name)
        assert hasattr(replaced, "_neuralops_metadata")
        assert replaced._neuralops_metadata[-1]["op"] == "replace"

        # Verify forward pass succeeds with correct dimensions
        x = torch.randn(2, 4, 32)
        out = replaced(x)
        assert out.shape == (2, 4, 32)
        assert torch.isfinite(out).all()

    def test_transplant_attention_with_calibrate(self):
        torch.manual_seed(42)
        model = DummyTransformerModel(num_layers=2, d_model=32)
        calib_inputs = torch.randn(2, 4, 32)
        transplanted = transplant_attention(
            model,
            attention="cosformer",
            calibrate=True,
            calibration_inputs=calib_inputs,
        )
        assert hasattr(transplanted, "_neuralops_metadata")

        # Verify forward pass succeeds post-calibration
        out = transplanted(calib_inputs)
        assert out.shape == (2, 4, 32)
        assert torch.isfinite(out).all()

    def test_replace_attention_none_model_raises(self):
        with pytest.raises(ValueError, match="model cannot be None"):
            replace_attention(None, attention="full_attention")

    def test_replace_attention_unknown_raises(self):
        from neuralops.core.exceptions import AttentionNotRegisteredError

        model = DummyTransformerModel(num_layers=1, d_model=32)
        with pytest.raises(AttentionNotRegisteredError):
            replace_attention(model, attention="non_existent_attention_mechanism_xyz")


class TestComposeAttentionAPI:
    """Verifies compose_attention modular factory."""

    def test_compose_attention_sdpa(self):
        torch.manual_seed(42)
        attn = compose_attention(
            projection="dense",
            positional="none",
            kernel_map="softmax",
            mask="causal",
            engine="sdpa",
            embed_dim=32,
            num_heads=4,
        )
        assert attn is not None
        x = torch.randn(2, 8, 32)
        out = attn(x)
        if isinstance(out, tuple):
            out = out[0]
        assert out.shape == (2, 8, 32)
        assert not torch.isnan(out).any()

    def test_compose_attention_linear_map(self):
        torch.manual_seed(42)
        attn = compose_attention(
            projection="dense",
            positional="none",
            kernel_map="relu",
            mask="none",
            engine="sdpa",
            embed_dim=32,
            num_heads=4,
        )
        x = torch.randn(2, 8, 32)
        out = attn(x)
        if isinstance(out, tuple):
            out = out[0]
        assert out.shape == (2, 8, 32)


class TestCompressAPI:
    """Verifies compress API and metadata tracking."""

    def test_compress_svd_metadata(self):
        model = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 32),
        )
        compressed = compress(model, method="svd", rank=16)
        assert hasattr(compressed, "_neuralops_metadata")
        assert compressed._neuralops_metadata[-1]["op"] == "compress"
        assert compressed._neuralops_metadata[-1]["type"] == "svd"
        assert hasattr(compressed, "_neuralops_compression_stats")
