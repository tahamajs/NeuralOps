"""Unit test for Triton & PyTorch fallback single-step recurrent update kernels."""
import pytest
import torch
from neuralops.core.triton_kernels.state_update import (
    fused_recurrent_linear_step,
    fused_recurrent_gla_step,
    fused_recurrent_gdn_step,
    fused_recurrent_hgrn2_step,
    causal_conv1d_step,
    fused_recurrent_ssm_step,
)


@pytest.mark.parametrize("device_str", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_fused_recurrent_linear_step(device_str):
    device = torch.device(device_str)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    B, H, Dk, Dv = 2, 4, 32, 32
    q_t = torch.randn(B, H, Dk, device=device, dtype=dtype)
    k_t = torch.randn(B, H, Dk, device=device, dtype=dtype)
    v_t = torch.randn(B, H, Dv, device=device, dtype=dtype)
    kv_state = torch.zeros(B, H, Dk, Dv, device=device, dtype=dtype)
    z_state = torch.zeros(B, H, Dk, device=device, dtype=dtype)

    o_t, next_kv, next_z = fused_recurrent_linear_step(q_t, k_t, v_t, kv_state, z_state)
    assert o_t.shape == (B, H, Dv)
    assert next_kv.shape == (B, H, Dk, Dv)
    assert next_z.shape == (B, H, Dk)
    assert not torch.isnan(o_t).any()


@pytest.mark.parametrize("device_str", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_fused_recurrent_gla_step(device_str):
    device = torch.device(device_str)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    B, H, Dk, Dv = 2, 4, 32, 32
    q_t = torch.randn(B, H, Dk, device=device, dtype=dtype)
    k_t = torch.randn(B, H, Dk, device=device, dtype=dtype)
    v_t = torch.randn(B, H, Dv, device=device, dtype=dtype)
    gk_t = -torch.rand(B, H, Dk, device=device, dtype=dtype)  # log decay
    kv_state = torch.zeros(B, H, Dk, Dv, device=device, dtype=dtype)

    o_t, next_kv = fused_recurrent_gla_step(q_t, k_t, v_t, gk_t, kv_state)
    assert o_t.shape == (B, H, Dv)
    assert next_kv.shape == (B, H, Dk, Dv)
    assert not torch.isnan(o_t).any()


@pytest.mark.parametrize("device_str", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_fused_recurrent_gdn_step(device_str):
    device = torch.device(device_str)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    B, H, Dk, Dv = 2, 4, 32, 32
    q_t = torch.randn(B, H, Dk, device=device, dtype=dtype)
    k_t = torch.randn(B, H, Dk, device=device, dtype=dtype)
    v_t = torch.randn(B, H, Dv, device=device, dtype=dtype)
    g_t = torch.full((B, H), 0.95, device=device, dtype=dtype)
    beta_t = torch.full((B, H), 0.5, device=device, dtype=dtype)
    kv_state = torch.zeros(B, H, Dk, Dv, device=device, dtype=dtype)

    o_t, next_kv = fused_recurrent_gdn_step(q_t, k_t, v_t, g_t, beta_t, kv_state)
    assert o_t.shape == (B, H, Dv)
    assert next_kv.shape == (B, H, Dk, Dv)
    assert not torch.isnan(o_t).any()


@pytest.mark.parametrize("device_str", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_fused_recurrent_hgrn2_step(device_str):
    device = torch.device(device_str)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    B, H, Dk, Dv = 2, 4, 32, 32
    q_t = torch.randn(B, H, Dk, device=device, dtype=dtype)
    k_t = torch.randn(B, H, Dk, device=device, dtype=dtype)
    v_t = torch.randn(B, H, Dv, device=device, dtype=dtype)
    f_t = torch.full((B, H, Dk), 0.9, device=device, dtype=dtype)
    kv_state = torch.zeros(B, H, Dk, Dv, device=device, dtype=dtype)

    o_t, next_kv = fused_recurrent_hgrn2_step(q_t, k_t, v_t, f_t, kv_state)
    assert o_t.shape == (B, H, Dv)
    assert next_kv.shape == (B, H, Dk, Dv)
    assert not torch.isnan(o_t).any()


@pytest.mark.parametrize("device_str", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_causal_conv1d_and_ssm_step(device_str):
    device = torch.device(device_str)
    dtype = torch.float32

    B, D_inner, D_state, K = 2, 32, 16, 4
    x_t = torch.randn(B, D_inner, device=device, dtype=dtype)
    conv_state = torch.zeros(B, D_inner, K - 1, device=device, dtype=dtype)
    conv_w = torch.randn(D_inner, 1, K, device=device, dtype=dtype)
    conv_b = torch.randn(D_inner, device=device, dtype=dtype)

    c_out, next_conv_state = causal_conv1d_step(x_t, conv_state, conv_w, conv_b)
    assert c_out.shape == (B, D_inner)
    assert next_conv_state.shape == (B, D_inner, K - 1)

    dt_t = torch.rand(B, D_inner, device=device, dtype=dtype)
    A = -torch.rand(D_inner, D_state, device=device, dtype=dtype)
    B_t = torch.randn(B, D_state, device=device, dtype=dtype)
    C_t = torch.randn(B, D_state, device=device, dtype=dtype)
    D_p = torch.ones(D_inner, device=device, dtype=dtype)
    ssm_state = torch.zeros(B, D_inner, D_state, device=device, dtype=dtype)

    y, next_ssm_state = fused_recurrent_ssm_step(c_out, dt_t, A, B_t, C_t, D_p, ssm_state)
    assert y.shape == (B, D_inner)
    assert next_ssm_state.shape == (B, D_inner, D_state)
    assert not torch.isnan(y).any()
