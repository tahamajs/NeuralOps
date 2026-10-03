"""
Comprehensive Unit Test Suite for Production-Hardened ARMOR Pruner.
Covers:
1. Factory config safety (None, dict, missing parameters)
2. Conv1D transposition & shape compatibility
3. LoRALinear wrapper preservation and adapter policy
4. Fallback behavior (max_error thresholding & fallback_to_original toggle)
5. Target attention only filtering (attention vs FFN / nested projections)
6. Multi-batch calibration DataLoader iteration
7. Numerical stability in FP16 / FP32 (no NaN / Inf)
8. Cross-architecture compatibility (DistilBERT, RoBERTa, GPT-2, Qwen3.5 dummy backbones)
"""

import sys
from pathlib import Path

# Add project Codes to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
import torch
import torch.nn as nn
from framework.compression_engine.armor_pruner import (
    ARMORLinear, ARMORPruner, armor_pruner_factory,
    compute_resilient_taylor_saliency)
from torch.utils.data import DataLoader, TensorDataset


class DummyConv1D(nn.Module):
    """Mimics transformers.pytorch_utils.Conv1D (weights are [in_features, out_features])."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(*size_out)


class DummyLoRALinear(nn.Module):
    """Mimics LoRALinear wrapper with base_layer and adapter branches."""

    def __init__(self, base_layer: nn.Module, r: int = 4):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        in_f = getattr(base_layer, "in_features", 16)
        out_f = getattr(base_layer, "out_features", 16)
        self.lora_A = nn.Parameter(torch.randn(r, in_f) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_f, r) * 0.01)
        self.scaling = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return base_out + lora_out


class DummyAttentionFFNModel(nn.Module):
    """Model with both attention projections and FFN / MLP projections."""

    def __init__(self, d: int = 16):
        super().__init__()
        self.d = d
        self.config = type("Config", (), {"hidden_size": d, "num_attention_heads": 2, "num_key_value_heads": 2})()
        self.attention = nn.ModuleDict({
            "q_proj": nn.Linear(d, d),
            "k_proj": nn.Linear(d, d),
            "v_proj": nn.Linear(d, d),
            "out_proj": nn.Linear(d, d),
        })
        self.ffn = nn.ModuleDict({
            "gate_proj": nn.Linear(d, d * 2),
            "up_proj": nn.Linear(d, d * 2),
            "down_proj": nn.Linear(d * 2, d),
            "lin1": nn.Linear(d, d),
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.attention["q_proj"](x)
        k = self.attention["k_proj"](x)
        v = self.attention["v_proj"](x)
        attn_out = self.attention["out_proj"](q + k + v)
        g = self.ffn["gate_proj"](attn_out)
        u = self.ffn["up_proj"](attn_out)
        return self.ffn["down_proj"](g * u)


def test_factory_config_safety():
    """Test 1: Factory with None, empty, or dict comp_config does not crash and respects overrides."""
    model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8))
    # 1.1 comp_config=None
    res = armor_pruner_factory(model, comp_config=None)
    assert res is not None

    # 1.2 comp_config as dictionary with specific overrides
    model2 = nn.Sequential(nn.Linear(16, 16))
    cfg_dict = {
        "armor_rank": 8,
        "armor_max_error": 0.20,
        "armor_target_attention_only": False,
        "armor_lora_alpha": 2.0,
    }
    res2, stats = armor_pruner_factory(model2, comp_config=cfg_dict, return_stats=True)
    assert stats["rank"] == 8
    assert stats["max_error"] == 0.20
    assert stats["lora_alpha"] == 2.0


def test_conv1d_transposition():
    """Test 2: Conv1D [in_features, out_features] is correctly transposed and replaced by ARMORLinear."""
    in_dim, out_dim = 16, 32
    conv1d = DummyConv1D(in_dim, out_dim)
    model = nn.Sequential(conv1d)

    compressor = ARMORPruner(model, rank=4, structured_24=True, decouple=True)
    compressed_model, stats = compressor.compress()

    replaced = list(compressed_model.children())[0]
    assert isinstance(replaced, ARMORLinear)
    assert replaced.in_features == in_dim
    assert replaced.out_features == out_dim
    assert replaced.sparse_weight.shape == (out_dim, in_dim)
    assert replaced.lora_A.shape == (4, in_dim)
    assert replaced.lora_B.shape == (out_dim, 4)

    # Test forward pass shape
    dummy_input = torch.randn(2, 5, in_dim)
    out = compressed_model(dummy_input)
    assert out.shape == (2, 5, out_dim)


def test_loralinear_wrapper_preservation():
    """Test 3: Wrapper is preserved, base_layer is replaced, wrapper LoRA zeroed, ARMOR LoRA trainable."""
    base = nn.Linear(16, 16)
    wrapper = DummyLoRALinear(base, r=4)
    model = nn.Sequential(wrapper)

    compressor = ARMORPruner(model, rank=4, max_error=0.8, decouple=True)
    compressed_model, stats = compressor.compress()

    curr_wrapper = list(compressed_model.children())[0]
    assert isinstance(curr_wrapper, DummyLoRALinear)
    assert isinstance(curr_wrapper.base_layer, ARMORLinear)

    # Outer LoRA weights zeroed and frozen to prevent double correction
    assert torch.all(curr_wrapper.lora_A == 0.0)
    assert torch.all(curr_wrapper.lora_B == 0.0)
    assert not curr_wrapper.lora_A.requires_grad
    assert not curr_wrapper.lora_B.requires_grad

    # ARMORLinear inner LoRA is active and trainable
    assert curr_wrapper.base_layer.lora_A.requires_grad
    assert curr_wrapper.base_layer.lora_B.requires_grad
    # Sparse base weight is frozen
    assert not curr_wrapper.base_layer.sparse_weight.requires_grad


def test_fallback_behavior():
    """Test 4: Fallback triggered when error > max_error with parameter freezing."""
    model = nn.Sequential(nn.Linear(16, 16))
    # 4.1 fallback_to_original=True with impossible max_error=0.0001
    comp = ARMORPruner(model, max_error=0.0001, fallback_to_original=True)
    comp.compress()
    assert len(comp.fallback_layers) == 1
    assert len(comp.compressed_layers) == 0
    # Original linear retained
    assert isinstance(list(model.children())[0], nn.Linear)

    # 4.2 fallback_to_original=False: layer replaced despite high error
    model2 = nn.Sequential(nn.Linear(16, 16))
    comp2 = ARMORPruner(model2, max_error=0.0001, fallback_to_original=False)
    comp2.compress()
    assert len(comp2.compressed_layers) == 1
    assert isinstance(list(model2.children())[0], ARMORLinear)


def test_target_attention_only():
    """Test 5: target_attention_only=True compresses only attention projections, strictly skipping FFN."""
    model = DummyAttentionFFNModel(d=16)
    comp = ARMORPruner(model, target_attention_only=True, rank=4, max_error=0.8)
    comp.compress()

    # Attention projections should be ARMORLinear
    for k in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        assert isinstance(model.attention[k], ARMORLinear), f"{k} should be ARMORLinear"

    # FFN layers must remain standard nn.Linear
    for k in ["gate_proj", "up_proj", "down_proj", "lin1"]:
        assert isinstance(model.ffn[k], nn.Linear), f"{k} should remain nn.Linear"


def test_calibration_with_dataloader():
    """Test 6: Calibration loader iterates over all batches and aggregates activations."""
    model = nn.Sequential(nn.Linear(16, 16))
    # Create DataLoader with 2 batches of size 3 and 4 (total 7 samples)
    data = torch.randn(7, 16)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=4)

    comp = ARMORPruner(model, calibration_loader=loader)
    act_map = comp._collect_activations()

    # The layer key should be "0"
    assert "0" in act_map
    # Total samples collected should equal total samples across batches
    assert act_map["0"].shape[0] == 7


def test_numerical_stability():
    """Test 7: FP16 tensors do not produce NaN/Inf in parameters or forward outputs."""
    model = nn.Sequential(nn.Linear(16, 16).half())
    comp = ARMORPruner(model, rank=4, max_error=0.8)
    comp.compress()

    replaced = list(model.children())[0]
    assert isinstance(replaced, ARMORLinear)
    assert not torch.isnan(replaced.sparse_weight).any()
    assert not torch.isnan(replaced.lora_A).any()
    assert not torch.isnan(replaced.lora_B).any()

    # Forward check with half input
    x = torch.randn(2, 16).half()
    out = model(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_architecture_compatibility():
    """Test 8: Compatibility across dummy models representing diverse transformer families."""
    # 8.1 BERT/RoBERTa style
    class DummyBert(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(32, 32)
            self.key = nn.Linear(32, 32)
            self.value = nn.Linear(32, 32)
            self.dense = nn.Linear(32, 32)
        def forward(self, x):
            return self.dense(self.query(x) + self.key(x) + self.value(x))

    bert_dummy = DummyBert()
    comp_bert = ARMORPruner(bert_dummy, target_attention_only=True, max_error=0.8)
    _, stats_bert = comp_bert.compress()
    assert stats_bert["num_compressed"] > 0

    # 8.2 GPT style (Conv1D c_attn, c_proj)
    class DummyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = DummyConv1D(32, 96)
            self.c_proj = DummyConv1D(96, 32)
        def forward(self, x):
            return self.c_proj(self.c_attn(x))

    gpt_dummy = DummyGPT()
    comp_gpt = ARMORPruner(gpt_dummy, target_attention_only=True, max_error=0.8)
    _, stats_gpt = comp_gpt.compress()
    assert stats_gpt["num_compressed"] == 2

    # 8.3 Qwen style (q_proj, k_proj, v_proj, o_proj) with GQA asymmetric check
    class DummyQwen(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("QwenConfig", (), {"num_attention_heads": 8, "num_key_value_heads": 2, "hidden_size": 32})()
            self.q_proj = nn.Linear(32, 64)
            self.k_proj = nn.Linear(32, 16)
            self.v_proj = nn.Linear(32, 16)
            self.o_proj = nn.Linear(64, 32)
        def forward(self, x):
            return self.o_proj(self.q_proj(x))

    qwen_dummy = DummyQwen()
    # When skip_attention_qkv=True, asymmetric q/k/v should be skipped, only o_proj compressed
    comp_qwen = ARMORPruner(qwen_dummy, target_attention_only=True, skip_attention_qkv=True, max_error=0.8)
    _, stats_qwen = comp_qwen.compress()
    assert stats_qwen["num_compressed"] == 1  # Only o_proj
