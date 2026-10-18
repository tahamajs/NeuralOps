import types

import pytest
import torch
import torch.nn as nn
from neuralops.compression_engine.cur_compressor import (
    CURCompressor, CURLinear, compute_energy_scores, compute_leverage_scores,
    cur_compressor_factory)


class MockConv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.randn(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias


class MockLoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=16):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.scaling = alpha / r
        in_f = getattr(base_layer, "in_features", getattr(base_layer, "weight").shape[0])
        out_f = getattr(base_layer, "out_features", getattr(base_layer, "weight").shape[1])
        self.lora_A = nn.Parameter(torch.randn(r, in_f) * 0.1)
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))

    def forward(self, x):
        out = self.base_layer(x)
        lora_out = torch.matmul(torch.matmul(x, self.lora_A.t()), self.lora_B.t()) * self.scaling
        return out + lora_out


def test_sampling_scores():
    W = torch.randn(64, 128)
    # Leverage sampling
    col_lev, row_lev = compute_leverage_scores(W, rank=16)
    assert col_lev.shape == (128,)
    assert row_lev.shape == (64,)
    assert torch.isclose(col_lev.sum(), torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(row_lev.sum(), torch.tensor(1.0), atol=1e-4)

    # Energy sampling
    col_en, row_en = compute_energy_scores(W)
    assert col_en.shape == (128,)
    assert row_en.shape == (64,)
    assert torch.isclose(col_en.sum(), torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(row_en.sum(), torch.tensor(1.0), atol=1e-4)


def test_cur_linear_forward_matches_formula():
    in_f, out_f, r = 128, 64, 16
    lin = nn.Linear(in_f, out_f)
    cur = CURLinear.from_linear(lin, rank=r, method="leverage", deterministic=True)

    x = torch.randn(2, 5, in_f)
    y_forward = cur(x)

    W_eff = cur.get_weight()
    assert W_eff.shape == (out_f, in_f)
    y_expected = torch.matmul(x, W_eff.t()) + cur.bias
    assert torch.allclose(y_forward, y_expected, atol=1e-4)


def test_cur_linear_conv1d_transposition():
    nx, nf, r = 128, 64, 16
    conv = MockConv1D(nf=nf, nx=nx)
    cur = CURLinear.from_linear(conv, rank=r)
    assert cur.in_features == nx
    assert cur.out_features == nf

    x = torch.randn(2, 4, nx)
    out = cur(x)
    assert out.shape == (2, 4, nf)


def test_cur_compressor_lora_preservation():
    lin = nn.Linear(128, 64)
    lora_mod = MockLoRALinear(lin, r=4)
    model = nn.Sequential(lora_mod)

    compressor = CURCompressor(model, rank=16, max_error=1.0)
    compressed_model, stats = compressor.compress()

    # Base layer should be CURLinear, outer LoRA preserved and trainable
    assert isinstance(compressed_model[0].base_layer, CURLinear)
    assert compressed_model[0].lora_A.requires_grad is True
    assert compressed_model[0].lora_B.requires_grad is True


def test_cur_compressor_target_attention_only():
    class DummyTransformerLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(128, 128)
            self.k_proj = nn.Linear(128, 128)
            self.mlp = nn.Sequential(
                nn.Linear(128, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
            )

    layer = DummyTransformerLayer()
    compressor = CURCompressor(layer, rank=16, target_attention_only=True, max_error=1.0)
    compressed, stats = compressor.compress()

    # Attention projections compressed
    assert any("q_proj" in name for name in stats["compressed_layers"])
    assert any("k_proj" in name for name in stats["compressed_layers"])
    # MLP not compressed and frozen
    assert not any("mlp" in name for name in stats["compressed_layers"])
    for p in layer.mlp.parameters():
        assert p.requires_grad is False


def test_cur_compressor_fallback_and_freeze():
    lin = nn.Linear(128, 64)
    model = nn.Sequential(lin)

    # Force fallback with max_error = 0.0001
    compressor = CURCompressor(model, rank=8, max_error=0.0001, fallback_to_original=True)
    compressed, stats = compressor.compress()

    assert len(stats["fallback_layers"]) > 0
    # Fallback layer must have requires_grad=False
    for p in compressed.parameters():
        assert p.requires_grad is False


def test_cur_compressor_factory_configs():
    model = nn.Sequential(nn.Linear(128, 64))
    # Config is None
    m1 = cur_compressor_factory(model, comp_config=None, rank=16)
    assert m1 is not None

    # Config is dict
    cfg_dict = {"cur_rank": 16, "method": "energy", "target_attention_only": False}
    m2, stats = cur_compressor_factory(nn.Sequential(nn.Linear(128, 64)), comp_config=cfg_dict, return_stats=True)
    assert "effective_parameter_ratio" in stats
    assert stats["cur_method"] == "energy"
