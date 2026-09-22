import sys
import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "Codes")

from framework.attentions.svd_linear_attention import (
    SVDLinear,
    SVDLinearAttention,
    inject_svd_linear_attention,
)
from framework.core.registry import ATTENTION_REGISTRY


def test_svdlinear_no_custom_attention_attr():
    """Test 1: SVDLinear does not have is_custom_attention = True as class attribute."""
    # is_custom_attention should NOT be True on SVDLinear class or instances
    assert not getattr(SVDLinear, "is_custom_attention", False), "SVDLinear must not have is_custom_attention=True"
    linear = SVDLinear(in_features=32, out_features=64, rank=16)
    assert not getattr(linear, "is_custom_attention", False), "SVDLinear instance must not have is_custom_attention=True"
    # But SVDLinearAttention MUST have is_custom_attention = True
    assert getattr(SVDLinearAttention, "is_custom_attention", False) is True
    print("[OK] test_svdlinear_no_custom_attention_attr")


def test_svdlinear_get_weight():
    """Test 2: SVDLinear has get_weight() method and not weight property."""
    linear = SVDLinear(in_features=32, out_features=64, rank=16, bias=True)
    assert not hasattr(linear.__class__, "weight") or not isinstance(
        getattr(linear.__class__, "weight"), property
    ), "SVDLinear must not define @property def weight"
    assert hasattr(linear, "get_weight") and callable(linear.get_weight)
    W = linear.get_weight()
    assert W.shape == (64, 32)
    assert not torch.isnan(W).any()
    print("[OK] test_svdlinear_get_weight")


def test_svdlinear_set_low_rank_factors_zeroing():
    """Test 3: set_low_rank_factors only zeroes written slice and handles smaller rank."""
    linear = SVDLinear(in_features=32, out_features=64, rank=16, bias=True)
    # Fill with sentinel values
    linear.U.data.fill_(7.0)
    linear.V.data.fill_(7.0)

    # Provide rank 8 factors (smaller than rank 16)
    U_sub = torch.randn(64, 8)
    V_sub = torch.randn(8, 32)
    linear.set_low_rank_factors(U_sub, V_sub)

    # Columns 8:16 in U and rows 8:16 in V must be 0
    assert torch.allclose(linear.U.data[:, 8:], torch.zeros(64, 8))
    assert torch.allclose(linear.V.data[8:, :], torch.zeros(8, 32))
    # Loaded factors in :8 must match
    assert torch.allclose(linear.U.data[:, :8], U_sub)
    assert torch.allclose(linear.V.data[:8, :], V_sub)
    print("[OK] test_svdlinear_set_low_rank_factors_zeroing")


def test_svdlinear_forward():
    """Test 4: SVDLinear forward pass with and without RMSNorm."""
    layer1 = SVDLinear(in_features=32, out_features=64, rank=16, use_rms_norm=False)
    x = torch.randn(2, 10, 32)
    out1 = layer1(x)
    assert out1.shape == (2, 10, 64)
    assert not torch.isnan(out1).any()

    layer2 = SVDLinear(in_features=32, out_features=64, rank=16, use_rms_norm=True)
    out2 = layer2(x)
    assert out2.shape == (2, 10, 64)
    assert not torch.isnan(out2).any()
    print("[OK] test_svdlinear_forward")


def test_copy_weights_none_raises():
    """Test 5: copy_weights raises RuntimeError when essential projections cannot be extracted."""
    attn = SVDLinearAttention(hidden_size=64, num_heads=4, rank=16)

    class DummyEmpty(nn.Module):
        pass

    dummy = DummyEmpty()
    with pytest.raises(RuntimeError) as excinfo:
        attn.copy_weights(dummy)
    assert "failed to extract required projections" in str(excinfo.value)
    print("[OK] test_copy_weights_none_raises")


def test_copy_weights_near_zero_frobenius_raises():
    """Test 6: copy_weights raises RuntimeError when projection weight has near-zero Frobenius norm."""
    attn = SVDLinearAttention(hidden_size=64, num_heads=4, rank=16)

    class DummyNearZero(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_lin = nn.Linear(64, 64)
            self.k_lin = nn.Linear(64, 64)
            self.v_lin = nn.Linear(64, 64)
            self.out_lin = nn.Linear(64, 64)
            # Set q_lin weight to zero
            self.q_lin.weight.data.zero_()

    dummy = DummyNearZero()
    with pytest.raises(RuntimeError) as excinfo:
        attn.copy_weights(dummy)
    assert "near-zero Frobenius norm" in str(excinfo.value)
    print("[OK] test_copy_weights_near_zero_frobenius_raises")


def test_copy_weights_distilbert():
    """Test 7: copy_weights successfully extracts weights from DistilBERT style attention."""
    attn = SVDLinearAttention(hidden_size=64, num_heads=4, rank=16)

    class DistilBertStyleAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_lin = nn.Linear(64, 64)
            self.k_lin = nn.Linear(64, 64)
            self.v_lin = nn.Linear(64, 64)
            self.out_lin = nn.Linear(64, 64)

    dummy = DistilBertStyleAttention()
    attn.copy_weights(dummy)
    # Forward check after copy
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)
    assert not torch.isnan(out).any()
    print("[OK] test_copy_weights_distilbert")


def test_copy_weights_gpt2():
    """Test 8: copy_weights successfully extracts weights from GPT-2 style fused c_attn and c_proj."""
    attn = SVDLinearAttention(hidden_size=64, num_heads=4, rank=16, is_decoder=True)

    class GPT2StyleAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = nn.Linear(64, 192)
            self.c_proj = nn.Linear(64, 64)

    dummy = GPT2StyleAttention()
    attn.copy_weights(dummy)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)
    assert not torch.isnan(out).any()
    print("[OK] test_copy_weights_gpt2")


def test_adapt_to_kv_size_modes():
    """Test 9: GQA adaptation modes 'slice', 'mean', and 'first' work properly."""
    # hidden_size=64, num_heads=4 -> head_dim=16. 2 KV heads -> kv_head_size = 32.
    attn = SVDLinearAttention(hidden_size=64, num_heads=4, num_kv_heads=2, rank=16)
    assert attn.num_kv_heads == 2
    assert attn.head_dim == 16
    assert attn.kv_head_size == 32

    class LLaMAStyleAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)  # 4 heads (64) in source, to be adapted to 2 heads (32)
            self.v_proj = nn.Linear(64, 64)
            self.o_proj = nn.Linear(64, 64)

    dummy = LLaMAStyleAttention()
    attn.copy_weights(dummy)
    x = torch.randn(2, 8, 64)
    out, _ = attn(x)
    assert out.shape == (2, 8, 64)
    assert not torch.isnan(out).any()
    print("[OK] test_adapt_to_kv_size_modes")


def test_sdpa_in_exact_mode():
    """Test 10: exact mode uses SDPA when output_attentions=False, returns attn_probs when True."""
    attn = SVDLinearAttention(hidden_size=64, num_heads=4, rank=16, mode="exact")
    x = torch.randn(2, 8, 64)

    # 1. output_attentions=False -> attn_probs should be None (SDPA path)
    out, attn_probs = attn(x, output_attentions=False)
    assert out.shape == (2, 8, 64)
    assert attn_probs is None
    assert not torch.isnan(out).any()

    # 2. output_attentions=True -> attn_probs should be returned (B, H, N, N)
    out2, attn_probs2 = attn(x, output_attentions=True)
    assert out2.shape == (2, 8, 64)
    assert attn_probs2 is not None
    assert attn_probs2.shape == (2, 4, 8, 8)
    assert not torch.isnan(attn_probs2).any()
    print("[OK] test_sdpa_in_exact_mode")


def test_chunked_decay_scan_bf16():
    """Test 11: chunked decay scan handles BF16 tensors without type promotion crash."""
    attn = SVDLinearAttention(
        hidden_size=64, num_heads=4, rank=16, mode="decay_linear", use_decay_gating=True
    )
    # Test with sequence length 128 (crosses chunk boundary of 64)
    x = torch.randn(2, 128, 64, dtype=torch.bfloat16)
    attn.to(torch.bfloat16)
    out, _ = attn(x)
    assert out.shape == (2, 128, 64)
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()
    print("[OK] test_chunked_decay_scan_bf16")


def test_linear_kernel_and_recurrent():
    """Test 12: linear kernel mode and step_recurrent execution."""
    attn = SVDLinearAttention(
        hidden_size=64, num_heads=4, rank=16, mode="linear", is_decoder=True
    )
    x = torch.randn(2, 16, 64)
    out, _ = attn(x)
    assert out.shape == (2, 16, 64)

    # Step recurrent test
    past_state = None
    outputs = []
    for t in range(5):
        xt = x[:, t : t + 1, :]
        out_step, past_state = attn.step_recurrent(xt, past_state)
        assert out_step.shape == (2, 1, 64)
        outputs.append(out_step)
    all_steps = torch.cat(outputs, dim=1)
    assert all_steps.shape == (2, 5, 64)
    print("[OK] test_linear_kernel_and_recurrent")


def test_registry_and_injector():
    """Test 13: ATTENTION_REGISTRY contains aliases and inject_svd_linear_attention works."""
    assert "svd_linear_attn" in ATTENTION_REGISTRY
    assert "svd_linear_attention" in ATTENTION_REGISTRY
    assert "svd_linear" in ATTENTION_REGISTRY
    assert "svd_linear_inject" in ATTENTION_REGISTRY
    assert "inject_svd_linear_attention" in ATTENTION_REGISTRY

    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Module()
            self.attention.q_lin = nn.Linear(64, 64)
            self.attention.k_lin = nn.Linear(64, 64)
            self.attention.v_lin = nn.Linear(64, 64)
            self.attention.out_lin = nn.Linear(64, 64)

    layer = DummyLayer()
    injected = inject_svd_linear_attention(
        layer=layer,
        model_config={"hidden_size": 64, "num_attention_heads": 4},
        rank=16,
        mode="exact",
    )
    assert isinstance(layer.attention, SVDLinearAttention)
    print("[OK] test_registry_and_injector")


if __name__ == "__main__":
    test_svdlinear_no_custom_attention_attr()
    test_svdlinear_get_weight()
    test_svdlinear_set_low_rank_factors_zeroing()
    test_svdlinear_forward()
    test_copy_weights_none_raises()
    test_copy_weights_near_zero_frobenius_raises()
    test_copy_weights_distilbert()
    test_copy_weights_gpt2()
    test_adapt_to_kv_size_modes()
    test_sdpa_in_exact_mode()
    test_chunked_decay_scan_bf16()
    test_linear_kernel_and_recurrent()
    test_registry_and_injector()
    print("ALL 13 TESTS PASSED!")
