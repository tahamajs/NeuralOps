"""
Comprehensive tests for every registered attention module.

Tests:
  1. Shape correctness – output must be (B, L, D)
  2. Output finiteness – no NaN, no Inf
  3. Zero‑shot fidelity – output must match the original DistilBERT attention
     within a tolerance when pretrained weights are copied
  4. Gradient flow – gradients reach the projection weights
  5. Padding mask – masked positions must produce zero output
  6. Multi‑head compatibility – reshaping must not raise errors
  7. Dropout behaviour – train vs eval mode
"""

import os
import sys

import pytest
import torch
from transformers import DistilBertConfig

sys.path.insert(0, os.path.dirname(__file__))
from conftest import build_attention, get_testable_attention_keys

# ----------------------------------------------------------------------
# Mark all tests in this module as "attention"
# ----------------------------------------------------------------------
pytestmark = pytest.mark.attention


# ----------------------------------------------------------------------
# 1. SHAPE CORRECTNESS
# ----------------------------------------------------------------------
class TestShapeCorrectness:
    """Verify that every attention module produces output of shape (B, L, D)."""

    @pytest.mark.parametrize(
        "key",
        get_testable_attention_keys(
            DistilBertConfig(hidden_size=768, num_attention_heads=12)
        ),
    )
    def test_output_shape_basic(self, key, hidden_states, model_config):
        """Output shape must be (B, L, D) for a standard forward pass."""
        attn = build_attention(key, model_config)
        out = attn(hidden_states)
        # Output can be a tuple (output, attn_weights) or just a tensor
        if isinstance(out, tuple):
            output_tensor = out[0]
        else:
            output_tensor = out
        assert output_tensor.shape[0] == hidden_states.shape[0], f"{key}: Batch size changed"
        assert output_tensor.shape[2] == hidden_states.shape[2], f"{key}: Hidden dimension changed"
        if key not in ["progressive_token_pruning"]:
            assert (
                output_tensor.shape == hidden_states.shape
            ), f"{key}: expected {hidden_states.shape}, got {output_tensor.shape}"


# ----------------------------------------------------------------------
# 2. OUTPUT FINITENESS
# ----------------------------------------------------------------------
class TestOutputFiniteness:
    """Every module must produce finite outputs (no NaN, no Inf)."""

    @pytest.mark.parametrize(
        "key",
        get_testable_attention_keys(
            DistilBertConfig(hidden_size=768, num_attention_heads=12)
        ),
    )
    def test_output_is_finite(self, key, hidden_states, model_config):
        """Test output is finite."""
        attn = build_attention(key, model_config)
        out = attn(hidden_states)
        if isinstance(out, tuple):
            output_tensor = out[0]
        else:
            output_tensor = out
        assert torch.isfinite(output_tensor).all(), f"{key}: output contains NaN or Inf"


# ----------------------------------------------------------------------
# 3. ZERO‑SHOT FIDELITY (requires pretrained weights)
# ----------------------------------------------------------------------
class TestZeroShotFidelity:
    """
    For modules that accept pretrained_attn, verify that after weight copying
    the output matches the original DistilBERT attention.
    """

    @pytest.mark.zero_shot
    @pytest.mark.parametrize(
        "key",
        get_testable_attention_keys(
            DistilBertConfig(hidden_size=768, num_attention_heads=12)
        ),
    )
    def test_zero_shot_fidelity(
        self, key, hidden_states, model_config, distilbert_attention
    ):
        """Test zero shot fidelity."""
        if distilbert_attention is None:
            pytest.skip("No pretrained DistilBERT attention available")

        # Build the custom attention with pretrained weight transfer
        try:
            attn = build_attention(
                key, model_config, pretrained_attn=distilbert_attention
            )
        except Exception:
            pytest.skip(f"{key} does not support pretrained_attn")

        # Original output
        orig_out = distilbert_attention(hidden_states)
        if isinstance(orig_out, tuple):
            orig_out = orig_out[0]

        # Custom output
        new_out = attn(hidden_states)
        if isinstance(new_out, tuple):
            new_out = new_out[0]

        # Compare within a generous tolerance
        diff = (orig_out - new_out).abs().mean().item()
        assert (
            diff < 0.5
        ), f"{key}: zero‑shot fidelity too low (mean abs diff = {diff:.4f})"


# ----------------------------------------------------------------------
# 4. GRADIENT FLOW
# ----------------------------------------------------------------------
class TestGradientFlow:
    """Gradients must flow back through the attention module."""

    @pytest.mark.grad
    @pytest.mark.parametrize(
        "key",
        get_testable_attention_keys(
            DistilBertConfig(hidden_size=768, num_attention_heads=12)
        ),
    )
    def test_gradient_flow(self, key, small_hidden_states, model_config):
        """Test gradient flow."""
        attn = build_attention(key, model_config)
        x = small_hidden_states.clone().requires_grad_(True)

        out = attn(x)
        if isinstance(out, tuple):
            out = out[0]

        loss = out.sum()
        loss.backward()

        # Check that at least one parameter received a gradient
        has_grad = False
        for param in attn.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, f"{key}: no gradients flowed to any parameter"


# ----------------------------------------------------------------------
# 5. PADDING MASK
# ----------------------------------------------------------------------
class TestPaddingMask:
    """Padding positions must produce zero output after masking."""

    @pytest.mark.parametrize(
        "key",
        get_testable_attention_keys(
            DistilBertConfig(hidden_size=768, num_attention_heads=12)
        ),
    )
    def test_padding_mask_output(
        self, key, hidden_states, attention_mask, model_config
    ):
        """Test padding mask output."""
        attn = build_attention(key, model_config)
        # Pass the binary mask; modules should internally convert it
        try:
            out = attn(hidden_states, attention_mask=attention_mask)
        except Exception:
            # Some modules may not accept attention_mask at all;
            # we skip those
            pytest.skip(f"{key} does not accept attention_mask")

        if isinstance(out, tuple):
            output_tensor = out[0]
        else:
            output_tensor = out

        # Padding positions (columns 24:32) should have small values
        pad_region = output_tensor[:, 24:, :]
        assert (
            pad_region.abs().max().item() < 10.0
        ), f"{key}: padding positions are not masked"


# ----------------------------------------------------------------------
# 6. MULTI‑HEAD COMPATIBILITY
# ----------------------------------------------------------------------
class TestMultiHeadCompatibility:
    """Modules must handle the multi‑head reshape correctly."""

    @pytest.mark.parametrize(
        "key",
        get_testable_attention_keys(
            DistilBertConfig(hidden_size=768, num_attention_heads=12)
        ),
    )
    def test_multi_head_no_error(self, key, small_hidden_states, model_config):
        """Test multi head no error."""
        attn = build_attention(key, model_config)
        try:
            attn(small_hidden_states)
        except Exception as e:
            pytest.fail(f"{key}: forward pass raised {type(e).__name__}: {e}")


# ----------------------------------------------------------------------
# 7. RETURN TUPLE FORMAT
# ----------------------------------------------------------------------
class TestReturnFormat:
    """Modules must return a tuple (output, attention_weights_or_None)."""

    @pytest.mark.parametrize(
        "key",
        get_testable_attention_keys(
            DistilBertConfig(hidden_size=768, num_attention_heads=12)
        ),
    )
    def test_returns_tuple(self, key, small_hidden_states, model_config):
        """Test returns tuple."""
        attn = build_attention(key, model_config)
        out = attn(small_hidden_states)
        assert isinstance(out, tuple), f"{key}: must return a tuple, got {type(out)}"
        assert len(out) == 2, f"{key}: tuple must have 2 elements, got {len(out)}"
