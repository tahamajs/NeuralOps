"""Subagent 6: Qwen3.5-0.8B Dedicated Test Suite (DeltaNet, GQA Attention, MoE)."""
import pytest
import torch
import torch.nn as nn
from framework.core.architecture import get_attention_modules
from framework.core.registry import is_module_declared_compatible
from framework.utils.lora import find_all_linear_submodule_names
from framework.utils.model_family import ModelAdapter, detect_family


class MockQwen3_5DeltaNet(nn.Module):
    def __init__(self, hidden=32, qk_dim=16, v_dim=16):
        super().__init__()
        self.qk_proj = nn.Linear(hidden, qk_dim)
        self.v_proj = nn.Linear(hidden, v_dim)
        self.out_proj = nn.Linear(v_dim, hidden)
    def forward(self, x, **kwargs):
        return self.out_proj(self.v_proj(x)), None

class MockQwen3_5GatedAttention(nn.Module):
    def __init__(self, hidden=32, q_dim=32, kv_dim=16):
        super().__init__()
        self.q_proj = nn.Linear(hidden, q_dim)
        self.k_proj = nn.Linear(hidden, kv_dim)
        self.v_proj = nn.Linear(hidden, kv_dim)
        self.o_proj = nn.Linear(q_dim, hidden)
    def forward(self, x, **kwargs):
        return self.o_proj(self.q_proj(x)), None

class MockQwen3_5MoEBlock(nn.Module):
    def __init__(self, hidden=32, num_experts=4, expert_dim=64):
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, expert_dim), nn.SiLU(), nn.Linear(expert_dim, hidden))
            for _ in range(num_experts)
        ])
    def forward(self, x):
        scores = torch.softmax(self.gate(x), dim=-1)
        out = sum(scores[..., i:i+1] * exp(x) for i, exp in enumerate(self.experts))
        return out

class MockQwen3_5FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {
            "model_type": "qwen3_5",
            "hidden_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_hidden_layers": 2,
        })()
        self.layer0_deltanet = MockQwen3_5DeltaNet()
        self.layer0_moe = MockQwen3_5MoEBlock()
        self.layer1_attn = MockQwen3_5GatedAttention()
        self.layer1_moe = MockQwen3_5MoEBlock()
        self.classifier = nn.Linear(32, 2)

    def forward(self, x):
        h, _ = self.layer0_deltanet(x)
        h = h + self.layer0_moe(h)
        h, _ = self.layer1_attn(h)
        h = h + self.layer1_moe(h)
        return self.classifier(h)

def test_qwen3_5_family_detection():
    m = MockQwen3_5FullModel()
    assert detect_family(m) == "qwen3_5"
    adapter = ModelAdapter(m)
    assert adapter.causal is True
    assert adapter.bidirectional is False
    assert adapter.position_embedding == "rotary"

def test_qwen3_5_leaf_discovery():
    m = MockQwen3_5FullModel()
    leaves = get_attention_modules(m)
    assert len(leaves) == 2
    assert any("deltanet" in name for name, _ in leaves)
    assert any("attn" in name for name, _ in leaves)

def test_qwen3_5_lora_dynamic_discovery():
    m = MockQwen3_5FullModel()
    targets = find_all_linear_submodule_names(m)
    assert "qk_proj" in targets
    assert "q_proj" in targets
    assert "gate" in targets
    assert "classifier" not in targets  # Head excluded

def test_qwen3_5_capability_checks():
    is_compat_causal, _ = is_module_declared_compatible("adaptive_span", "qwen3_5")
    assert is_compat_causal is True

    is_compat_svd, _ = is_module_declared_compatible("adaptive_svd", "qwen3_5")
    assert is_compat_svd is True

def test_qwen3_5_forward_backward():
    m = MockQwen3_5FullModel()
    x = torch.randn(2, 4, 32, requires_grad=True)
    out = m(x)
    assert out.shape == (2, 4, 2)
    assert torch.isfinite(out).all()

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
