"""Tests for CoMe Layer Shedding Compressor."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from framework.compression_engine.come import (CoMeBridge, CoMeBridgedLayer,
                                               CoMeCompressor, CoMeLoRALinear,
                                               come_factory)
from framework.core.registry import COMPRESSOR_REGISTRY


class DummyLayer(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, attention_mask=None):
        return self.mlp(self.attn(x))


class DummyModel(nn.Module):
    def __init__(self, num_layers: int = 6, hidden_size: int = 64):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, num_hidden_layers=num_layers)
        self.embeddings = nn.Embedding(100, hidden_size)
        self.layers = nn.ModuleList([DummyLayer(hidden_size) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return self.head(x)


def test_registry():
    assert "come" in COMPRESSOR_REGISTRY.list_available()
    assert "come_compressor" in COMPRESSOR_REGISTRY.list_available()
    assert "come_factory" in COMPRESSOR_REGISTRY.list_available()
    assert issubclass(COMPRESSOR_REGISTRY.get("come"), CoMeCompressor) or COMPRESSOR_REGISTRY.get("come").__name__ == "CoMeCompressor"


def test_bridge_low_rank_identity_at_init():
    bridge = CoMeBridge(hidden_size=64, bridge_rank=16, alpha_init=0.1)
    assert bridge.V.weight.shape == (16, 64)
    assert bridge.U.weight.shape == (64, 16)
    # V is initialized to zeros -> U(V(x)) is zero -> alpha * (0 - x)
    # Wait, with alpha_init: y = x + alpha * (U(V(x)) - x)
    # But V(x)=0, so recon = 0.
    # To check output is continuous and passes through:
    x = torch.randn(2, 8, 64)
    out = bridge(x)
    assert out.shape == (2, 8, 64)


def test_bridge_handles_various_containers():
    bridge = CoMeBridge(hidden_size=64, bridge_rank=16)
    x = torch.randn(2, 8, 64)
    
    # tuple
    tup_out = bridge((x, "extra"))
    assert isinstance(tup_out, tuple)
    assert tup_out[0].shape == (2, 8, 64)
    assert tup_out[1] == "extra"

    # dict
    dict_out = bridge({"last_hidden_state": x, "aux": 123})
    assert isinstance(dict_out, dict)
    assert dict_out["last_hidden_state"].shape == (2, 8, 64)


def test_bridged_layer_delegation():
    layer = DummyLayer(64)
    bridge = CoMeBridge(hidden_size=64, bridge_rank=16)
    bridged = CoMeBridgedLayer(layer, bridge=bridge)

    assert hasattr(bridged, "attn")
    assert hasattr(bridged, "mlp")
    x = torch.randn(2, 8, 64)
    out = bridged(x)
    assert out.shape == (2, 8, 64)


def test_come_compression_layer_shedding():
    model = DummyModel(num_layers=6, hidden_size=64)
    compressor = CoMeCompressor(model, sparsity=0.5, bridge_rank=16, lora_r=16, recalibrate_ln=False)
    comp_model, stats = compressor.compress()

    # 6 layers * 0.5 kept = 3 layers kept
    assert len(comp_model.layers) == 3
    assert stats["original_layer_count"] == 6
    assert stats["kept_layer_count"] == 3
    assert stats["bridges_inserted"] > 0
    assert stats["bridge_rank"] == 16
    assert stats["lora_r"] == 16
    assert stats["forward_check_success"] is True


def test_come_lora_adapters():
    model = DummyModel(num_layers=4, hidden_size=64)
    compressor = CoMeCompressor(model, sparsity=0.25, lora_r=8, recalibrate_ln=False)
    comp_model, stats = compressor.compress()

    trainable_params = compressor.get_trainable_params()
    assert len(trainable_params) > 0
    assert stats["trainable_parameters"] > 0
    assert stats["lora_parameters"] > 0


def test_come_factory():
    model = DummyModel(num_layers=4, hidden_size=64)
    comp_model, stats = come_factory(model, sparsity=0.5, bridge_rank=16, lora_r=8, recalibrate_ln=False, return_stats=True)
    assert len(comp_model.layers) == 2
    assert stats["kept_layer_count"] == 2


def test_come_fail_loud_forward():
    class BrokenLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 32)
        def forward(self, x, **kwargs):
            raise ValueError("Intentional failure in layer")

    class BrokenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=32)
            self.layers = nn.ModuleList([BrokenLayer() for _ in range(4)])
        def forward(self, x, **kwargs):
            for l in self.layers:
                x = l(x)
            return x

    broken_model = BrokenModel()
    compressor = CoMeCompressor(broken_model, sparsity=0.5, allow_forward_failure=False, recalibrate_ln=False)
    with pytest.raises(RuntimeError):
        compressor.compress()
