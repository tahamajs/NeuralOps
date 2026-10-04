"""Unit tests for weight copying, QKV slicing, and projection detection."""

import pytest
import torch
import torch.nn as nn
from neuralops.core.adapters import (CANONICAL_PROJECTION_ALIASES,
                                     UniversalAdapter)
from neuralops.core.attention_utils import (_copy_projection_weights,
                                            _slice_fused_qkv_weights,
                                            compute_reconstruction_error)
from neuralops.utils.model_family import ModelAdapter


@pytest.fixture(autouse=True)
def set_random_seed():
    torch.manual_seed(42)


class TestSliceFusedQKVWeights:
    """Verify _slice_fused_qkv_weights handles Conv1D and Linear fused QKV projections."""

    def test_conv1d_transposition_and_slicing(self):
        # GPT-2 Conv1D weight: shape [in_features, 3 * out_features] -> [32, 96]
        in_dim = 32
        out_dim = 32
        weight = torch.randn(in_dim, 3 * out_dim)
        bias = torch.randn(3 * out_dim)

        sliced = _slice_fused_qkv_weights(weight, bias, num_heads=2)
        q_w, q_b = sliced["q"]
        k_w, k_b = sliced["k"]
        v_w, v_b = sliced["v"]

        # Linear expects [out_features, in_features] -> [32, 32]
        assert q_w.shape == (out_dim, in_dim)
        assert k_w.shape == (out_dim, in_dim)
        assert v_w.shape == (out_dim, in_dim)
        assert q_b.shape == (out_dim,)
        assert k_b.shape == (out_dim,)
        assert v_b.shape == (out_dim,)

        # Exact values check
        expected_q = weight[:, :out_dim].t().contiguous()
        assert torch.allclose(q_w, expected_q, atol=1e-6)

    def test_standard_linear_fused_slicing(self):
        # Standard Linear fused QKV: shape [3 * out_features, in_features] -> [96, 32]
        in_dim = 32
        out_dim = 32
        weight = torch.randn(3 * out_dim, in_dim)
        bias = torch.randn(3 * out_dim)

        sliced = _slice_fused_qkv_weights(weight, bias, num_heads=2)
        q_w, q_b = sliced["q"]
        assert q_w.shape == (out_dim, in_dim)
        assert torch.allclose(q_w, weight[:out_dim, :], atol=1e-6)
        assert torch.allclose(q_b, bias[:out_dim], atol=1e-6)

    def test_missing_bias_handling(self):
        weight = torch.randn(32, 96)
        sliced = _slice_fused_qkv_weights(weight, None, num_heads=2)
        assert sliced["q"][1] is None
        assert sliced["k"][1] is None
        assert sliced["v"][1] is None


class TestCopyProjectionWeights:
    """Verify _copy_projection_weights across all module conventions and safeguards."""

    @pytest.mark.parametrize("proj_name", [
        "q_proj", "q_lin", "query", "k_proj", "k_lin", "key",
        "v_proj", "v_lin", "value", "out_proj", "out_lin", "o_proj", "dense",
    ])
    def test_standard_linear_copy(self, proj_name):
        class Mod(nn.Module):
            def __init__(self):
                super().__init__()
                setattr(self, proj_name, nn.Linear(32, 32))

        mod = Mod()
        src_w = torch.randn(32, 32)
        src_b = torch.randn(32)

        _copy_projection_weights(getattr(mod, proj_name), src_w, src_b)
        target = getattr(mod, proj_name)
        assert torch.allclose(target.weight, src_w, atol=1e-6)
        assert torch.allclose(target.bias, src_b, atol=1e-6)

        err = compute_reconstruction_error(src_w, target)
        assert err < 1e-5

    def test_property_target_protection(self):
        """Verify that layers with @property weight are safely handled without crash."""
        class FactorizedPropertyMock(nn.Module):
            def __init__(self):
                super().__init__()
                self.U = nn.Parameter(torch.randn(32, 8))
                self.V = nn.Parameter(torch.randn(8, 32))
                self.bias = nn.Parameter(torch.zeros(32))

            @property
            def weight(self):
                return self.U @ self.V

        mod = FactorizedPropertyMock()
        src_w = torch.randn(32, 32)
        src_b = torch.randn(32)

        # Must not raise AttributeError: can't set attribute
        _copy_projection_weights(mod, src_w, src_b)

    def test_graceful_missing_projection_fallback(self):
        """Target module with unexpected structure logs warning and does not crash."""
        class CustomEmpty(nn.Module):
            pass

        mod = CustomEmpty()
        src_w = torch.randn(32, 32)
        _copy_projection_weights(mod, src_w, None)


class TestUniversalAdapterAndModelAdapter:
    """Verify get_weight_bias retrieves weights correctly across naming conventions."""

    def test_universal_adapter_linear(self):
        class AttentionBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Linear(32, 32)
                self.key = nn.Linear(32, 32)
                self.value = nn.Linear(32, 32)
                self.dense = nn.Linear(32, 32, bias=False)

        block = AttentionBlock()
        q_w, q_b = UniversalAdapter.get_weight_bias(block, "q")
        assert q_w.shape == (32, 32)
        assert q_b is not None

        out_w, out_b = UniversalAdapter.get_weight_bias(block, "o")
        assert out_w.shape == (32, 32)
        assert out_b is None

    def test_canonical_aliases_coverage(self):
        for role in ("q", "k", "v", "o"):
            assert role in CANONICAL_PROJECTION_ALIASES
            aliases = CANONICAL_PROJECTION_ALIASES[role]
            assert len(aliases) >= 3
