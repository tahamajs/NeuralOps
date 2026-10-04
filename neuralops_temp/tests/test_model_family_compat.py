import types

import pytest
import torch
import torch.nn as nn
from neuralops.utils.compat import (FamilyModuleWrapper, convert_mask,
                                    fp32_linalg, install_projection_aliases,
                                    module_family_support, wrap_for_family)
from neuralops.utils.model_family import ModelAdapter, detect_family


def cfg(model_type, hidden_size=896, heads=14, head_dim=64, layers=24):
    return types.SimpleNamespace(model_type=model_type, hidden_size=hidden_size,
                                 num_attention_heads=heads, head_dim=head_dim,
                                 num_hidden_layers=layers)


@pytest.mark.parametrize("model_type,expected", [
    ("distilbert", "bert"), ("bert", "bert"), ("roberta", "roberta"),
    ("gpt2", "gpt"), ("qwen2", "qwen"), ("t5", "t5"),
    ("mamba", "ssm"), ("madeup", "unknown"),
])
def test_family_detection(model_type, expected):
    assert detect_family(cfg(model_type)) == expected


def test_adapter_metadata():
    q = ModelAdapter(cfg("qwen2", 768, 12, 64, 6), "qwen")
    assert (q.hidden_size, q.num_heads, q.head_dim, q.num_hidden_layers) == (768, 12, 64, 6)
    assert q.causal is True and q.position_embedding == "rotary"
    b = ModelAdapter(cfg("bert", 768, 12, 64, 6), "bert")
    assert b.causal is False and b.position_embedding == "absolute"


def test_projection_aliases():
    class Projections(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4)
            self.k_proj = nn.Linear(4, 4)
            self.v_proj = nn.Linear(4, 4)
            self.o_proj = nn.Linear(4, 4)
    m = Projections()
    install_projection_aliases(m)
    assert m.q_lin is m.q_proj and m.k_lin is m.k_proj
    assert m.v_lin is m.v_proj and m.out_lin is m.o_proj
    plain = nn.Linear(4, 4)
    install_projection_aliases(plain)
    assert not hasattr(plain, "q_lin")


def test_mask_conversion():
    adapter = ModelAdapter(cfg("qwen2"), "qwen")
    mask = torch.tensor([[[[0.0, float("-inf")], [0.0, 0.0]]]])
    converted = convert_mask(mask, adapter, expects_bidirectional=True)
    assert converted.dtype == torch.bool and converted.shape == mask.shape
    assert torch.equal(converted, mask >= 0)
    assert convert_mask(mask, adapter, expects_bidirectional=False) is mask


def test_fp32_linalg():
    u, s, vt = fp32_linalg(torch.linalg.svd, torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16))
    assert u.dtype == s.dtype == vt.dtype == torch.float16
    assert u.shape == (2, 2) and s.shape == (2,) and vt.shape == (2, 2)
    assert torch.allclose((u.float() @ torch.diag(s.float()) @ vt.float()), torch.tensor([[1., 2.], [3., 4.]]), atol=2e-2)


def test_support_registry():
    assert module_family_support("mamba")["qwen"] == "incompatible"
    assert module_family_support("mamba")["bert"] == "encoder_compatible"
    assert module_family_support("faq")["qwen"] == "adapted"


def test_wrap_for_family():
    class DummyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(4, 4)
            self.k_proj = nn.Linear(4, 4)
            self.v_proj = nn.Linear(4, 4)
            self.o_proj = nn.Linear(4, 4)
        def forward(self, x, **kwargs):
            return self.o_proj(self.v_proj(x))
    bert = DummyAttention()
    assert wrap_for_family(bert, "bert") is bert
    qwen = DummyAttention()
    wrapped = wrap_for_family(qwen, "qwen")
    assert isinstance(wrapped, FamilyModuleWrapper)
    assert wrapped(torch.randn(2, 3, 4)).shape == (2, 3, 4)
