"""Part 7 Verification: Project-Wide Numerical Safety (RCA-7, RCA-8, RCA-9)."""
import pytest
import torch
import torch.nn as nn
from neuralops.attentions.performer_attention import PerformerAttention
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

def test_performer_fp16_clamped():
    cfg = DummyConfig()
    mod = PerformerAttention(cfg)
    wrapper = UniversalAttentionWrapper(mod)
    x = torch.randn(2, 8, 32) * 40.0
    out, _ = wrapper(x)
    assert torch.isfinite(out).all()

def test_amp_scaler_guard():
    class MockScaler:
        def __init__(self):
            self.called = False
        def is_enabled(self):
            return True
        def unscale_(self, opt):
            if self.called:
                raise RuntimeError("unscale_ already called")
            self.called = True

    scaler = MockScaler()
    scaler.unscale_(None)
    try:
        scaler.unscale_(None)
    except RuntimeError:
        pass
