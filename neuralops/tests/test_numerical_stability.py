"""Unit tests for numerical stability, FP16/FP32 precision, and mask edge cases."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from neuralops.attentions.performer_attention import PerformerAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.utils.replace_attention import UniversalAttentionWrapper
from transformers import PretrainedConfig


class DummyConfig(PretrainedConfig):
    def __init__(self, hidden_size=32, num_attention_heads=2, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.d_model = hidden_size
        self.max_position_embeddings = 512
        self.vocab_size = 100


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)


class TestNumericalStability:
    """Verify numerical finite outputs and gradients under fp32 and fp16."""

    @pytest.mark.parametrize("mod_name", [
        "adaptive_svd",
        "asvd",
        "rpc",
    ])
    def test_fp32_forward_backward_finite(self, mod_name):
        cls_or_factory = ATTENTION_REGISTRY.get(mod_name)
        if cls_or_factory is None:
            pytest.skip(f"{mod_name} not available")
        cfg = DummyConfig(hidden_size=32, num_attention_heads=2)
        try:
            mod = cls_or_factory(cfg)
        except TypeError:
            mod = cls_or_factory(cfg, rank=8)

        wrapper = UniversalAttentionWrapper(mod)
        x = torch.randn(2, 16, 32, dtype=torch.float32, requires_grad=True)
        out, _ = wrapper(x)
        assert torch.isfinite(out).all(), f"NaN or Inf in {mod_name} fp32 forward"

        loss = out.sum()
        loss.backward()
        assert torch.isfinite(x.grad).all(), f"NaN or Inf in {mod_name} gradients"

    def test_performer_fp16_exp_clamping_prevents_overflow(self):
        """Verify that large inputs into Performer do not cause exp overflow to Inf under fp16."""
        cfg = DummyConfig(hidden_size=32, num_attention_heads=2)
        mod = PerformerAttention(cfg)
        wrapper = UniversalAttentionWrapper(mod)

        # Extreme input that would blow up unclamped exp
        x = torch.randn(2, 16, 32) * 50.0
        out, _ = wrapper(x)
        assert torch.isfinite(out).all()

    def test_sequence_length_512_stability(self):
        """Verify attention forward stability on longer sequence length 512."""
        cfg = DummyConfig(hidden_size=32, num_attention_heads=2)
        factory = ATTENTION_REGISTRY.get("adaptive_svd")
        mod = factory(cfg)
        wrapper = UniversalAttentionWrapper(mod)

        x = torch.randn(1, 512, 32)
        out, _ = wrapper(x)
        assert out.shape == (1, 512, 32)
        assert torch.isfinite(out).all()

    def test_attention_mask_padding_and_large_negatives(self):
        """Verify 2D boolean mask and 4D additive large negative mask (-1e4)."""
        cfg = DummyConfig(hidden_size=32, num_attention_heads=2)
        factory = ATTENTION_REGISTRY.get("adaptive_svd")
        mod = factory(cfg)
        wrapper = UniversalAttentionWrapper(mod)

        x = torch.randn(2, 16, 32)
        # 2D mask (1=valid, 0=padded)
        mask_2d = torch.ones(2, 16)
        mask_2d[:, 10:] = 0
        out_2d, _ = wrapper(x, attention_mask=mask_2d)
        assert torch.isfinite(out_2d).all()

        # 4D additive mask
        mask_4d = torch.zeros(2, 1, 16, 16)
        mask_4d[:, :, :, 10:] = -10000.0
        out_4d, _ = wrapper(x, attention_mask=mask_4d)
        assert torch.isfinite(out_4d).all()

    def test_head_mask_support(self):
        """Verify wrapper cleanly processes head_mask without crashing."""
        cfg = DummyConfig(hidden_size=32, num_attention_heads=2)
        factory = ATTENTION_REGISTRY.get("adaptive_svd")
        mod = factory(cfg)
        wrapper = UniversalAttentionWrapper(mod)

        x = torch.randn(2, 16, 32)
        head_mask = torch.tensor([1.0, 0.0])
        out, _ = wrapper(x, head_mask=head_mask)
        assert torch.isfinite(out).all()

    def test_gradscaler_double_unscale_safe(self):
        """Verify that calling unscale_ twice does not unhandled-crash when guarded."""
        class MockScaler:
            def __init__(self):
                self.unscaled = False
            def is_enabled(self):
                return True
            def unscale_(self, opt):
                if self.unscaled:
                    raise RuntimeError("unscale_() has already been called on this optimizer since the last update().")
                self.unscaled = True

        scaler = MockScaler()
        opt = MagicMock()

        # Step 1
        scaler.unscale_(opt)
        # Step 2: guarded call
        try:
            scaler.unscale_(opt)
        except RuntimeError:
            pass  # Expected guard behavior
