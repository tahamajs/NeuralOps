# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Verification test for Silent Truncation TORCH_CHECK fixes (Part 1).
Tests both:
1. Direct CUDA host TORCH_CHECK exception on oversized dimensions.
2. Graceful fallback to Triton / PyTorch reference on oversized dimensions via autograd wrapper.
3. Numerical parity (diff < 1e-3) on valid dimensions.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def check_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")


def test_alibi_attention_silent_truncation():
    from neuralops.core.cuda_kernels.alibi_attention.alibi_autograd import (
        _get_module,
        alibi_attention,
        alibi_attention_pytorch_ref,
    )
    mod = _get_module()
    B, H, L = 2, 4, 32
    slopes = torch.randn(H, device="cuda", dtype=torch.float32)

    D_valid = 64
    q = torch.randn(B, H, L, D_valid, device="cuda", dtype=torch.float16) * 0.1
    k = torch.randn(B, H, L, D_valid, device="cuda", dtype=torch.float16) * 0.1
    v = torch.randn(B, H, L, D_valid, device="cuda", dtype=torch.float16) * 0.1
    out_valid = alibi_attention(q, k, v, slopes, is_causal=True)
    ref_valid = alibi_attention_pytorch_ref(q, k, v, slopes, is_causal=True)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    D_over = 160
    q_over = torch.randn(B, H, L, D_over, device="cuda", dtype=torch.float16) * 0.1
    k_over = torch.randn(B, H, L, D_over, device="cuda", dtype=torch.float16) * 0.1
    v_over = torch.randn(B, H, L, D_over, device="cuda", dtype=torch.float16) * 0.1

    with pytest.raises(RuntimeError, match="exceeds max 128"):
        mod.alibi_attention_fwd(q_over, k_over, v_over, slopes, True)

    out_fallback = alibi_attention(q_over, k_over, v_over, slopes, is_causal=True)
    ref_fallback = alibi_attention_pytorch_ref(q_over, k_over, v_over, slopes, is_causal=True)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


def test_cluster_attention_silent_truncation():
    from neuralops.core.cuda_kernels.cluster_attention.cluster_attention_autograd import (
        _get_module,
        cluster_attention,
        cluster_attention_ref,
    )
    mod = _get_module()
    B, H, N, K_clust = 2, 4, 32, 16

    D_valid = 64
    q = torch.randn(B, H, N, D_valid, device="cuda", dtype=torch.float16) * 0.1
    c_cen = torch.randn(B, H, K_clust, D_valid, device="cuda", dtype=torch.float16) * 0.1
    v = torch.randn(B, H, K_clust, D_valid, device="cuda", dtype=torch.float16) * 0.1
    c_idx = torch.randint(0, K_clust, (B, H, N), device="cuda", dtype=torch.int32)
    out_valid = cluster_attention(q, c_cen, v, c_idx)
    ref_valid = cluster_attention_ref(q, c_cen, v, c_idx)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    D_over = 160
    q_over = torch.randn(B, H, N, D_over, device="cuda", dtype=torch.float16) * 0.1
    c_cen_over = torch.randn(B, H, K_clust, D_over, device="cuda", dtype=torch.float16) * 0.1
    v_over = torch.randn(B, H, K_clust, D_over, device="cuda", dtype=torch.float16) * 0.1

    with pytest.raises(RuntimeError, match="exceeds max 128"):
        mod.cluster_attn_fwd(q_over, c_cen_over, v_over, c_idx)

    out_fallback = cluster_attention(q_over, c_cen_over, v_over, c_idx)
    ref_fallback = cluster_attention_ref(q_over, c_cen_over, v_over, c_idx)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


def test_cross_attention_silent_truncation():
    from neuralops.core.cuda_kernels.cross_attention.cross_attn_autograd import (
        _get_module,
        cross_attention,
        cross_attention_pytorch_ref,
    )
    mod = _get_module()
    B, H, L_q, L_kv = 2, 4, 32, 48

    D_valid = 64
    q = torch.randn(B, H, L_q, D_valid, device="cuda", dtype=torch.float16) * 0.1
    k = torch.randn(B, H, L_kv, D_valid, device="cuda", dtype=torch.float16) * 0.1
    v = torch.randn(B, H, L_kv, D_valid, device="cuda", dtype=torch.float16) * 0.1
    out_valid = cross_attention(q, k, v)
    ref_valid = cross_attention_pytorch_ref(q, k, v)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    D_over = 160
    q_over = torch.randn(B, H, L_q, D_over, device="cuda", dtype=torch.float16) * 0.1
    k_over = torch.randn(B, H, L_kv, D_over, device="cuda", dtype=torch.float16) * 0.1
    v_over = torch.randn(B, H, L_kv, D_over, device="cuda", dtype=torch.float16) * 0.1

    with pytest.raises(RuntimeError, match="exceeds max 128"):
        mod.cross_attention_fwd(q_over, k_over, v_over)

    out_fallback = cross_attention(q_over, k_over, v_over)
    ref_fallback = cross_attention_pytorch_ref(q_over, k_over, v_over)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


def test_differential_attention_silent_truncation():
    from neuralops.core.cuda_kernels.differential_attention.differential_attention_autograd import (
        _get_module,
        differential_attention,
        differential_attention_ref,
    )
    mod = _get_module()
    B, H, N = 2, 4, 32

    D_valid = 64
    q1 = torch.randn(B, H, N, D_valid, device="cuda", dtype=torch.float16) * 0.1
    k1 = torch.randn(B, H, N, D_valid, device="cuda", dtype=torch.float16) * 0.1
    q2 = torch.randn(B, H, N, D_valid, device="cuda", dtype=torch.float16) * 0.1
    k2 = torch.randn(B, H, N, D_valid, device="cuda", dtype=torch.float16) * 0.1
    v = torch.randn(B, H, N, D_valid, device="cuda", dtype=torch.float16) * 0.1
    out_valid = differential_attention(q1, k1, q2, k2, v, lambda_val=0.8)
    ref_valid = differential_attention_ref(q1, k1, q2, k2, v, lambda_val=0.8)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    D_over = 160
    q1_over = torch.randn(B, H, N, D_over, device="cuda", dtype=torch.float16) * 0.1
    k1_over = torch.randn(B, H, N, D_over, device="cuda", dtype=torch.float16) * 0.1
    q2_over = torch.randn(B, H, N, D_over, device="cuda", dtype=torch.float16) * 0.1
    k2_over = torch.randn(B, H, N, D_over, device="cuda", dtype=torch.float16) * 0.1
    v_over = torch.randn(B, H, N, D_over, device="cuda", dtype=torch.float16) * 0.1

    with pytest.raises(RuntimeError, match="exceeds max 128"):
        mod.diff_attn_fwd(q1_over, k1_over, q2_over, k2_over, v_over, 0.8)

    out_fallback = differential_attention(q1_over, k1_over, q2_over, k2_over, v_over, lambda_val=0.8)
    ref_fallback = differential_attention_ref(q1_over, k1_over, q2_over, k2_over, v_over, lambda_val=0.8)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


def test_dynamic_conv1d_silent_truncation():
    from neuralops.core.cuda_kernels.dynamic_conv1d.dynamic_conv1d_autograd import (
        _get_module,
        dynamic_conv1d,
        dynamic_conv1d_ref,
    )
    mod = _get_module()

    B, D, L = 2, 32, 64
    x_3d = torch.randn(B, D, L, device="cuda", dtype=torch.float16) * 0.1
    w_3d_valid = torch.randn(D, 8, device="cuda", dtype=torch.float16) * 0.1
    out_valid = dynamic_conv1d(x_3d, w_3d_valid)
    ref_valid = dynamic_conv1d_ref(x_3d, w_3d_valid)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    w_3d_over = torch.randn(D, 24, device="cuda", dtype=torch.float16) * 0.1
    with pytest.raises(RuntimeError, match="> 16 not supported"):
        mod.dynamic_conv1d_fwd(x_3d, w_3d_over, None)

    out_fallback_3d = dynamic_conv1d(x_3d, w_3d_over)
    ref_fallback_3d = dynamic_conv1d_ref(x_3d, w_3d_over)
    assert (out_fallback_3d - ref_fallback_3d).abs().max() < 1e-3

    B, H, T, N, K = 2, 4, 32, 32, 4
    x_4d = torch.randn(B, H, T + K - 1, 64, device="cuda", dtype=torch.float16) * 0.1
    w_4d = torch.randn(B, H, N, K, device="cuda", dtype=torch.float16) * 0.1
    out_4d_valid = dynamic_conv1d(x_4d, w_4d)
    ref_4d_valid = dynamic_conv1d_ref(x_4d, w_4d)
    assert (out_4d_valid - ref_4d_valid).abs().max() < 1e-3

    x_4d_over = torch.randn(B, H, T + K - 1, 160, device="cuda", dtype=torch.float16) * 0.1
    with pytest.raises(RuntimeError, match="> 128 not supported"):
        mod.dynamic_conv1d_fwd(x_4d_over, w_4d, None)

    out_fallback_4d = dynamic_conv1d(x_4d_over, w_4d)
    ref_fallback_4d = dynamic_conv1d_ref(x_4d_over, w_4d)
    assert (out_fallback_4d - ref_fallback_4d).abs().max() < 1e-3


def test_entropy_gate_silent_truncation():
    from neuralops.core.cuda_kernels.entropy_gate.entropy_gate_autograd import (
        _get_module,
        spectral_entropy_svd_gate,
        spectral_entropy_svd_gate_pytorch_ref,
    )
    mod = _get_module()
    B, M, K_dim, N_dim, R = 2, 32, 64, 64, 160
    x = torch.randn(B, M, K_dim, device="cuda", dtype=torch.float16) * 0.1
    V_r = torch.randn(K_dim, R, device="cuda", dtype=torch.float16) * 0.1
    U_r = torch.randn(R, N_dim, device="cuda", dtype=torch.float16) * 0.1

    eff_r_valid = 32
    out_valid = spectral_entropy_svd_gate(x, V_r, U_r, eff_r_valid)
    ref_valid = spectral_entropy_svd_gate_pytorch_ref(x, V_r, U_r, eff_r_valid)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    eff_r_over = 140
    with pytest.raises(RuntimeError, match="> 128 not supported"):
        mod.spectral_entropy_svd_gate_fwd(x, V_r, U_r, eff_r_over, None, None)

    out_fallback = spectral_entropy_svd_gate(x, V_r, U_r, eff_r_over)
    ref_fallback = spectral_entropy_svd_gate_pytorch_ref(x, V_r, U_r, eff_r_over)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


def test_flash_decoding_silent_truncation():
    from neuralops.core.cuda_kernels.flash_decoding.flash_dec_autograd import (
        _get_module,
        flash_decoding_attention,
        flash_decoding_attention_pytorch_ref,
    )
    mod = _get_module()
    B, H, N_q, N_kv = 2, 4, 1, 64

    D_valid = 128
    q = torch.randn(B, H, N_q, D_valid, device="cuda", dtype=torch.float16) * 0.1
    k = torch.randn(B, H, N_kv, D_valid, device="cuda", dtype=torch.float16) * 0.1
    v = torch.randn(B, H, N_kv, D_valid, device="cuda", dtype=torch.float16) * 0.1
    out_valid = flash_decoding_attention(q, k, v)
    ref_valid = flash_decoding_attention_pytorch_ref(q, k, v)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    D_over = 320
    q_over = torch.randn(B, H, N_q, D_over, device="cuda", dtype=torch.float16) * 0.1
    k_over = torch.randn(B, H, N_kv, D_over, device="cuda", dtype=torch.float16) * 0.1
    v_over = torch.randn(B, H, N_kv, D_over, device="cuda", dtype=torch.float16) * 0.1

    with pytest.raises(RuntimeError, match="> 256 not supported"):
        mod.flash_decoding_attention_fwd(q_over, k_over, v_over)

    out_fallback = flash_decoding_attention(q_over, k_over, v_over)
    ref_fallback = flash_decoding_attention_pytorch_ref(q_over, k_over, v_over)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


def test_fms_scan_silent_truncation():
    from neuralops.core.cuda_kernels.fms_scan.fms_autograd import (
        _get_module,
        fms_scan,
        fms_scan_pytorch_ref,
    )
    mod = _get_module()
    B, H, L = 2, 4, 32
    decay = torch.full((H,), 0.95, device="cuda", dtype=torch.float32)

    D_valid = 64
    q = torch.randn(B, H, L, D_valid, device="cuda", dtype=torch.float16) * 0.1
    k = torch.randn(B, H, L, D_valid, device="cuda", dtype=torch.float16) * 0.1
    v = torch.randn(B, H, L, D_valid, device="cuda", dtype=torch.float16) * 0.1
    out_valid = fms_scan(q, k, v, decay)
    ref_valid = fms_scan_pytorch_ref(q, k, v, decay)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    D_over = 160
    q_over = torch.randn(B, H, L, D_over, device="cuda", dtype=torch.float16) * 0.1
    k_over = torch.randn(B, H, L, D_over, device="cuda", dtype=torch.float16) * 0.1
    v_over = torch.randn(B, H, L, D_over, device="cuda", dtype=torch.float16) * 0.1

    with pytest.raises(RuntimeError, match="> 128 not supported"):
        mod.fms_scan_fwd(q_over, k_over, v_over, decay)

    out_fallback = fms_scan(q_over, k_over, v_over, decay)
    ref_fallback = fms_scan_pytorch_ref(q_over, k_over, v_over, decay)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


def test_gla_scan_silent_truncation():
    from neuralops.core.cuda_kernels.gla_scan.gla_autograd import (
        _get_module,
        gla_scan,
        gla_scan_pytorch_ref,
    )
    mod = _get_module()
    B, H, L = 2, 4, 32

    D_valid = 64
    q = torch.randn(B, H, L, D_valid, device="cuda", dtype=torch.float16) * 0.1
    k = torch.randn(B, H, L, D_valid, device="cuda", dtype=torch.float16) * 0.1
    v = torch.randn(B, H, L, D_valid, device="cuda", dtype=torch.float16) * 0.1
    out_valid = gla_scan(q, k, v)
    ref_valid = gla_scan_pytorch_ref(q, k, v)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    D_over = 160
    q_over = torch.randn(B, H, L, D_over, device="cuda", dtype=torch.float16) * 0.1
    k_over = torch.randn(B, H, L, D_over, device="cuda", dtype=torch.float16) * 0.1
    v_over = torch.randn(B, H, L, D_over, device="cuda", dtype=torch.float16) * 0.1

    with pytest.raises(RuntimeError, match="> 128 not supported"):
        mod.gla_scan_fwd(q_over, k_over, v_over, None)

    out_fallback = gla_scan(q_over, k_over, v_over)
    ref_fallback = gla_scan_pytorch_ref(q_over, k_over, v_over)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


def test_latent_kv_attn_silent_truncation():
    from neuralops.core.cuda_kernels.latent_kv_attn.latent_kv_autograd import (
        _get_module,
        latent_kv_cache_attention,
        latent_kv_cache_attention_pytorch_ref,
    )
    mod = _get_module()
    B, H, L_Q, L_KV, D_K, D_V = 2, 4, 32, 48, 64, 64

    R_valid = 32
    q = torch.randn(B, H, L_Q, D_K, device="cuda", dtype=torch.float16) * 0.1
    c_kv = torch.randn(B, L_KV, R_valid, device="cuda", dtype=torch.float16) * 0.1
    u_k = torch.randn(D_K, R_valid, device="cuda", dtype=torch.float16) * 0.1
    u_v = torch.randn(R_valid, D_V, device="cuda", dtype=torch.float16) * 0.1
    out_valid = latent_kv_cache_attention(q, c_kv, u_k, u_v)
    ref_valid = latent_kv_cache_attention_pytorch_ref(q, c_kv, u_k, u_v)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    R_over = 160
    c_kv_over = torch.randn(B, L_KV, R_over, device="cuda", dtype=torch.float16) * 0.1
    u_k_over = torch.randn(D_K, R_over, device="cuda", dtype=torch.float16) * 0.1
    u_v_over = torch.randn(R_over, D_V, device="cuda", dtype=torch.float16) * 0.1

    with pytest.raises(RuntimeError, match="> 128 not supported"):
        mod.latent_kv_cache_attention_fwd(q, c_kv_over, u_k_over, u_v_over, 1.0 / (R_over ** 0.5))

    out_fallback = latent_kv_cache_attention(q, c_kv_over, u_k_over, u_v_over)
    ref_fallback = latent_kv_cache_attention_pytorch_ref(q, c_kv_over, u_k_over, u_v_over)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


def test_linformer_attention_silent_truncation():
    from neuralops.core.cuda_kernels.linformer_attention.linformer_attention_autograd import (
        _get_module,
        linformer_attention,
        linformer_attention_ref,
    )
    mod = _get_module()
    B, H, N, D = 2, 4, 32, 64

    K_proj_valid = 32
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16) * 0.1
    k_proj = torch.randn(B, H, K_proj_valid, D, device="cuda", dtype=torch.float16) * 0.1
    v_proj = torch.randn(B, H, K_proj_valid, D, device="cuda", dtype=torch.float16) * 0.1
    out_valid = linformer_attention(q, k_proj, v_proj)
    ref_valid = linformer_attention_ref(q, k_proj, v_proj)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    K_proj_over = 160
    k_proj_over = torch.randn(B, H, K_proj_over, D, device="cuda", dtype=torch.float16) * 0.1
    v_proj_over = torch.randn(B, H, K_proj_over, D, device="cuda", dtype=torch.float16) * 0.1

    with pytest.raises(RuntimeError, match="> 128 not supported"):
        mod.linformer_attn_fwd(q, k_proj_over, v_proj_over)

    out_fallback = linformer_attention(q, k_proj_over, v_proj_over)
    ref_fallback = linformer_attention_ref(q, k_proj_over, v_proj_over)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


def test_local_attention_silent_truncation():
    from neuralops.core.cuda_kernels.local_attention.local_attention_autograd import (
        _get_module,
        local_attention,
        local_attention_ref,
    )
    mod = _get_module()
    B, H, N = 2, 4, 32

    D_valid = 64
    q = torch.randn(B, H, N, D_valid, device="cuda", dtype=torch.float16) * 0.1
    k = torch.randn(B, H, N, D_valid, device="cuda", dtype=torch.float16) * 0.1
    v = torch.randn(B, H, N, D_valid, device="cuda", dtype=torch.float16) * 0.1
    out_valid = local_attention(q, k, v, window_size=16)
    ref_valid = local_attention_ref(q, k, v, window_size=16)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    D_over = 160
    q_over = torch.randn(B, H, N, D_over, device="cuda", dtype=torch.float16) * 0.1
    k_over = torch.randn(B, H, N, D_over, device="cuda", dtype=torch.float16) * 0.1
    v_over = torch.randn(B, H, N, D_over, device="cuda", dtype=torch.float16) * 0.1

    with pytest.raises(RuntimeError, match="> 128 not supported"):
        mod.local_attn_fwd(q_over, k_over, v_over, 16)

    out_fallback = local_attention(q_over, k_over, v_over, window_size=16)
    ref_fallback = local_attention_ref(q_over, k_over, v_over, window_size=16)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3
