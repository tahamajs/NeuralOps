import pytest
import torch
import torch.nn as nn
from framework.compression_engine.dobi_svd import (
    DobiSVDLinearLayer,
    DobiSVDCompressor,
    dobi_factory,
    inject_dobi,
)


class DummyAttentionModel(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
        self.ffn = nn.Linear(d, 4 * d)
        self.classifier = nn.Linear(d, 2)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.matmul(q, k.transpose(-1, -2)) / 8.0
        out = torch.matmul(torch.softmax(attn, dim=-1), v)
        out = self.out_proj(out)
        return self.classifier(out)


class DummyConv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.randn(nx, nf) * 0.02)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(size_out)


def test_dobi_layer_basic_and_no_output_scale():
    layer = DobiSVDLinearLayer(in_features=64, out_features=128, rank=16)
    assert not hasattr(layer, "output_scale")
    assert not hasattr(layer, "__iter__") or type(layer).__iter__ is nn.Module.__iter__
    x = torch.randn(2, 8, 64)
    out = layer(x)
    assert out.shape == (2, 8, 128)


def test_dobi_layer_backward_gradients():
    layer = DobiSVDLinearLayer(in_features=64, out_features=64, rank=16)
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    assert layer.U.grad is not None
    assert layer.V.grad is not None
    assert layer.gate.grad is not None
    assert layer.bias.grad is not None


def test_dobi_compressor_target_attention_only():
    model = DummyAttentionModel(d=64)
    compressor = DobiSVDCompressor(model, rank=16, target_attention_only=True)
    compressed_model, stats = compressor.compress()
    # Attention projections should be compressed
    assert isinstance(compressed_model.query, DobiSVDLinearLayer)
    assert isinstance(compressed_model.key, DobiSVDLinearLayer)
    assert isinstance(compressed_model.value, DobiSVDLinearLayer)
    assert isinstance(compressed_model.out_proj, DobiSVDLinearLayer)
    # FFN and classifier should NOT be compressed
    assert not isinstance(compressed_model.ffn, DobiSVDLinearLayer)
    assert not isinstance(compressed_model.classifier, DobiSVDLinearLayer)


def test_dobi_compressor_rank_cap_128():
    linear = nn.Linear(512, 512)
    compressor = DobiSVDCompressor(linear, rank=300)
    # Target rank was requested as 300, but max_r must cap it at 128
    r = compressor._determine_rank(512, 512, 300)
    assert r <= 128


def test_dobi_compress_conv1d():
    class ModelWithConv1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = DummyConv1D(nf=192, nx=64)
            self.c_proj = DummyConv1D(nf=64, nx=64)

        def forward(self, x):
            return self.c_proj(self.c_attn(x))

    model = ModelWithConv1D()
    compressor = DobiSVDCompressor(model, rank=16, target_attention_only=True)
    compressed_model, stats = compressor.compress()
    assert isinstance(compressed_model.c_attn, DobiSVDLinearLayer)
    assert isinstance(compressed_model.c_proj, DobiSVDLinearLayer)
    x = torch.randn(2, 8, 64)
    out = compressed_model(x)
    assert out.shape == (2, 8, 64)


def test_dobi_factory_and_inject():
    model = DummyAttentionModel(d=64)
    res = dobi_factory(model, rank=8)
    assert res is not None

    single_linear = nn.Linear(64, 64)
    injected = inject_dobi(single_linear, rank=8)
    assert injected is not None
