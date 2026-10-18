"""Subagent 2: Linear Attention & Kernelization Family Test Suite."""
import pytest
import torch
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.utils.replace_attention import UniversalAttentionWrapper
from transformers import PretrainedConfig


class DummyConfig(PretrainedConfig):
    def __init__(self, hidden_size=32, num_attention_heads=2, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_attention_heads
        self.head_dim = 16
        self.d_model = hidden_size

@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)

@pytest.mark.parametrize("mod_name", [
    "performer", "performer_attention", "cosformer", "fourier_mixer", "nystrom"
])
def test_linear_attention_forward(mod_name):
    factory = ATTENTION_REGISTRY.get(mod_name)
    if factory is None:
        pytest.skip(f"{mod_name} not registered")
    cfg = DummyConfig()
    try:
        mod = factory(cfg)
    except TypeError:
        mod = factory(cfg, rank=8)

    wrapper = UniversalAttentionWrapper(mod)
    x = torch.randn(2, 8, 32)
    out, _ = wrapper(x)
    assert out.shape == (2, 8, 32)
    assert torch.isfinite(out).all()
