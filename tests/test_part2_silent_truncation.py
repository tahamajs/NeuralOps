"""
Verification test for Silent Truncation TORCH_CHECK fixes (Part 2).
Tests all 11 modules:
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


# 1. longformer
def test_longformer_silent_truncation():
    from neuralops.core.cuda_kernels.longformer.longformer_autograd import (
        _get_module,
        longformer,
        longformer_ref,
    )
    mod = _get_module()
    B, H, N = 2, 4, 32

    # Valid D <= 128
    D_valid = 64
    q = torch.randn(B, H, N, D_valid, device="cuda", dtype=torch.float32) * 0.1
    k = torch.randn(B, H, N, D_valid, device="cuda", dtype=torch.float32) * 0.1
    v = torch.randn(B, H, N, D_valid, device="cuda", dtype=torch.float32) * 0.1
    out_valid = longformer(q, k, v, 64, 4)
    ref_valid = longformer_ref(q, k, v, 64, 4)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    # Oversized D > 128
    D_over = 160
    q_over = torch.randn(B, H, N, D_over, device="cuda", dtype=torch.float32) * 0.1
    k_over = torch.randn(B, H, N, D_over, device="cuda", dtype=torch.float32) * 0.1
    v_over = torch.randn(B, H, N, D_over, device="cuda", dtype=torch.float32) * 0.1

    with pytest.raises(RuntimeError, match="longformer_fwd: D=.* > 128 not supported"):
        mod.longformer_fwd(q_over, k_over, v_over, 64, 4)

    out_fallback = longformer(q_over, k_over, v_over, 64, 4)
    ref_fallback = longformer_ref(q_over, k_over, v_over, 64, 4)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


# 2. mamba_svd_flash
def test_mamba_svd_flash_silent_truncation():
    from neuralops.core.cuda_kernels.mamba_svd_flash.mamba_svd_flash_autograd import (
        _get_module,
        mamba_svd_flash,
        mamba_svd_flash_ref,
    )
    mod = _get_module()
    B, N, D = 2, 32, 64

    # Valid R <= 128
    R_valid = 32
    u = torch.randn(B, N, D, device="cuda", dtype=torch.float32) * 0.1
    svd_u = torch.randn(D, R_valid, device="cuda", dtype=torch.float32) * 0.1
    svd_vt = torch.randn(R_valid, D, device="cuda", dtype=torch.float32) * 0.1
    decay = torch.sigmoid(torch.randn(B, N, D, device="cuda", dtype=torch.float32))

    out_valid = mamba_svd_flash(u, svd_u, svd_vt, decay)
    ref_valid = mamba_svd_flash_ref(u, svd_u, svd_vt, decay)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    # Oversized R > 128
    R_over = 160
    svd_u_over = torch.randn(D, R_over, device="cuda", dtype=torch.float32) * 0.1
    svd_vt_over = torch.randn(R_over, D, device="cuda", dtype=torch.float32) * 0.1

    with pytest.raises(RuntimeError, match="mamba_svd_flash_fwd: R=.* > 128 not supported"):
        mod.mamba_svd_flash_fwd(u.contiguous(), svd_u_over.contiguous(), svd_vt_over.contiguous(), decay.contiguous())

    out_fallback = mamba_svd_flash(u, svd_u_over, svd_vt_over, decay)
    ref_fallback = mamba_svd_flash_ref(u, svd_u_over, svd_vt_over, decay)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


# 3. multi_scale_retention
def test_multi_scale_retention_silent_truncation():
    from neuralops.core.cuda_kernels.multi_scale_retention.retention_autograd import (
        _get_module,
        multi_scale_retention,
        multi_scale_retention_pytorch_ref,
    )
    mod = _get_module()
    B, H, L = 2, 4, 32
    gammas = torch.tensor([0.9, 0.8, 0.7, 0.6], device="cuda", dtype=torch.float32)

    # Valid D <= 128
    D_valid = 64
    q = torch.randn(B, H, L, D_valid, device="cuda", dtype=torch.float32) * 0.1
    k = torch.randn(B, H, L, D_valid, device="cuda", dtype=torch.float32) * 0.1
    v = torch.randn(B, H, L, D_valid, device="cuda", dtype=torch.float32) * 0.1

    out_valid = multi_scale_retention(q, k, v, gammas)
    ref_valid = multi_scale_retention_pytorch_ref(q, k, v, gammas)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    # Oversized D > 128
    D_over = 160
    q_over = torch.randn(B, H, L, D_over, device="cuda", dtype=torch.float32) * 0.1
    k_over = torch.randn(B, H, L, D_over, device="cuda", dtype=torch.float32) * 0.1
    v_over = torch.randn(B, H, L, D_over, device="cuda", dtype=torch.float32) * 0.1

    with pytest.raises(RuntimeError, match="retention_scan_fwd: D=.* > 128 not supported"):
        mod.multi_scale_retention_fwd(q_over.contiguous(), k_over.contiguous(), v_over.contiguous(), gammas.contiguous())

    out_fallback = multi_scale_retention(q_over, k_over, v_over, gammas)
    ref_fallback = multi_scale_retention_pytorch_ref(q_over, k_over, v_over, gammas)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


# 4. nystrom_attention
def test_nystrom_attention_silent_truncation():
    from neuralops.core.cuda_kernels.nystrom_attention.nystrom_attention_autograd import (
        _get_module,
        nystrom_attention,
        nystrom_attention_ref,
    )
    mod = _get_module()
    B, H, N, D = 2, 4, 32, 64

    # Valid M <= 64
    M_valid = 32
    q_l = torch.randn(B, H, N, M_valid, device="cuda", dtype=torch.float32) * 0.1
    k_l = torch.randn(B, H, N, M_valid, device="cuda", dtype=torch.float32) * 0.1
    v_l = torch.randn(B, H, M_valid, D, device="cuda", dtype=torch.float32) * 0.1
    p_l = torch.randn(B, H, M_valid, M_valid, device="cuda", dtype=torch.float32) * 0.1

    out_valid = nystrom_attention(q_l, k_l, v_l, p_l)
    ref_valid = nystrom_attention_ref(q_l, k_l, v_l, p_l)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    # Oversized M > 64
    M_over = 80
    q_l_over = torch.randn(B, H, N, M_over, device="cuda", dtype=torch.float32) * 0.1
    k_l_over = torch.randn(B, H, N, M_over, device="cuda", dtype=torch.float32) * 0.1
    v_l_over = torch.randn(B, H, M_over, D, device="cuda", dtype=torch.float32) * 0.1
    p_l_over = torch.randn(B, H, M_over, M_over, device="cuda", dtype=torch.float32) * 0.1

    with pytest.raises(RuntimeError, match="nystrom_attention_fwd: M=.* > 64 not supported"):
        mod.nystrom_attn_fwd(q_l_over.contiguous(), k_l_over.contiguous(), v_l_over.contiguous(), p_l_over.contiguous())

    out_fallback = nystrom_attention(q_l_over, k_l_over, v_l_over, p_l_over)
    ref_fallback = nystrom_attention_ref(q_l_over, k_l_over, v_l_over, p_l_over)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


# 5. reformer
def test_reformer_silent_truncation():
    from neuralops.core.cuda_kernels.reformer.reformer_autograd import (
        _get_module,
        reformer,
        reformer_ref,
    )
    mod = _get_module()
    B, H, N = 2, 4, 32
    hb = torch.randint(0, 4, (B, N), device="cuda", dtype=torch.int32)

    # Valid D <= 128
    D_valid = 64
    qk = torch.randn(B, H, N, D_valid, device="cuda", dtype=torch.float32) * 0.1
    v = torch.randn(B, H, N, D_valid, device="cuda", dtype=torch.float32) * 0.1

    out_valid = reformer(qk, v, hb)
    ref_valid = reformer_ref(qk, v, hb)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    # Oversized D > 128
    D_over = 160
    qk_over = torch.randn(B, H, N, D_over, device="cuda", dtype=torch.float32) * 0.1
    v_over = torch.randn(B, H, N, D_over, device="cuda", dtype=torch.float32) * 0.1

    with pytest.raises(RuntimeError, match="reformer_fwd: D=.* > 128 not supported"):
        mod.reformer_fwd(qk_over.contiguous(), v_over.contiguous(), hb.contiguous().int())

    out_fallback = reformer(qk_over, v_over, hb)
    ref_fallback = reformer_ref(qk_over, v_over, hb)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


# 6. rotational_svd
def test_rotational_svd_silent_truncation():
    from neuralops.core.cuda_kernels.rotational_svd.rotational_svd_autograd import (
        _get_module,
        rotational_svd,
        rotational_svd_pytorch_ref,
    )
    mod = _get_module()
    M, K, N = 4, 64, 64
    x = torch.randn(M, K, device="cuda", dtype=torch.float32) * 0.1
    r_in = torch.randn(K, K, device="cuda", dtype=torch.float32) * 0.1

    # Valid R <= 64
    R_valid = 32
    v_r = torch.randn(R_valid, K, device="cuda", dtype=torch.float32) * 0.1
    u_r = torch.randn(N, R_valid, device="cuda", dtype=torch.float32) * 0.1

    out_valid = rotational_svd(x, r_in, v_r, u_r)
    ref_valid = rotational_svd_pytorch_ref(x, r_in, v_r, u_r)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    # Oversized R > 64
    R_over = 80
    v_r_over = torch.randn(R_over, K, device="cuda", dtype=torch.float32) * 0.1
    u_r_over = torch.randn(N, R_over, device="cuda", dtype=torch.float32) * 0.1

    with pytest.raises(RuntimeError, match="rotational_svd_fwd: R=.* > 64 not supported"):
        mod.rotational_svd_fwd(x.contiguous(), r_in.contiguous(), v_r_over.contiguous(), u_r_over.contiguous(), None)

    out_fallback = rotational_svd(x, r_in, v_r_over, u_r_over)
    ref_fallback = rotational_svd_pytorch_ref(x, r_in, v_r_over, u_r_over)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


# 7. routing_attention
def test_routing_attention_silent_truncation():
    from neuralops.core.cuda_kernels.routing_attention.routing_attention_autograd import (
        _get_module,
        routing_attention,
        routing_attention_ref,
    )
    mod = _get_module()
    B, H, N = 2, 4, 32
    rw = torch.rand(B, H, N, device="cuda", dtype=torch.float32)

    # Valid D <= 128
    D_valid = 64
    q = torch.randn(B, H, N, D_valid, device="cuda", dtype=torch.float32) * 0.1
    k = torch.randn(B, H, N, D_valid, device="cuda", dtype=torch.float32) * 0.1
    v = torch.randn(B, H, N, D_valid, device="cuda", dtype=torch.float32) * 0.1

    out_valid = routing_attention(q, k, v, rw)
    ref_valid = routing_attention_ref(q, k, v, rw)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    # Oversized D > 128
    D_over = 160
    q_over = torch.randn(B, H, N, D_over, device="cuda", dtype=torch.float32) * 0.1
    k_over = torch.randn(B, H, N, D_over, device="cuda", dtype=torch.float32) * 0.1
    v_over = torch.randn(B, H, N, D_over, device="cuda", dtype=torch.float32) * 0.1

    with pytest.raises(RuntimeError, match="routing_attention_fwd: D=.* > 128 not supported"):
        mod.routing_attn_fwd(q_over.contiguous(), k_over.contiguous(), v_over.contiguous(), rw.contiguous(), 32)

    out_fallback = routing_attention(q_over, k_over, v_over, rw)
    ref_fallback = routing_attention_ref(q_over, k_over, v_over, rw)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


# 8. rpca_sparse_svd
def test_rpca_sparse_svd_silent_truncation():
    from neuralops.core.cuda_kernels.rpca_sparse_svd.rpca_sparse_svd_autograd import (
        _get_module,
        rpca_sparse_svd_gemm,
        rpca_sparse_svd_gemm_pytorch_ref,
    )
    mod = _get_module()
    M, K, N = 4, 64, 64
    x = torch.randn(M, K, device="cuda", dtype=torch.float32) * 0.1

    # Valid R <= 128
    R_valid = 64
    v_r = torch.randn(K, R_valid, device="cuda", dtype=torch.float32) * 0.1
    u_r = torch.randn(R_valid, N, device="cuda", dtype=torch.float32) * 0.1

    out_valid = rpca_sparse_svd_gemm(x, v_r, u_r)
    ref_valid = rpca_sparse_svd_gemm_pytorch_ref(x, v_r, u_r)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    # Oversized R > 128
    R_over = 160
    v_r_over = torch.randn(K, R_over, device="cuda", dtype=torch.float32) * 0.1
    u_r_over = torch.randn(R_over, N, device="cuda", dtype=torch.float32) * 0.1

    with pytest.raises(RuntimeError, match="rpca_sparse_svd_fwd: R=.* > 128 not supported"):
        mod.rpca_sparse_svd_fwd(x.contiguous(), v_r_over.contiguous(), u_r_over.contiguous(), None, None)

    out_fallback = rpca_sparse_svd_gemm(x, v_r_over, u_r_over)
    ref_fallback = rpca_sparse_svd_gemm_pytorch_ref(x, v_r_over, u_r_over)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


# 9. selective_scan
def test_selective_scan_silent_truncation():
    from neuralops.core.cuda_kernels.selective_scan.selective_scan_autograd import (
        _get_module,
        selective_scan,
        selective_scan_pytorch_ref,
    )
    mod = _get_module()
    B_sz, D_sz, L_sz = 2, 16, 32
    u = torch.randn(B_sz, D_sz, L_sz, device="cuda", dtype=torch.float32) * 0.1
    delta = torch.randn(B_sz, D_sz, L_sz, device="cuda", dtype=torch.float32) * 0.1

    # Valid D_STATE <= 128
    D_state_valid = 16
    A_valid = -torch.rand(D_sz, D_state_valid, device="cuda", dtype=torch.float32)
    B_valid = torch.randn(B_sz, D_state_valid, L_sz, device="cuda", dtype=torch.float32) * 0.1
    C_valid = torch.randn(B_sz, D_state_valid, L_sz, device="cuda", dtype=torch.float32) * 0.1

    out_valid, _ = selective_scan(u, delta, A_valid, B_valid, C_valid)
    ref_valid, _ = selective_scan_pytorch_ref(u, delta, A_valid, B_valid, C_valid)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    # Oversized D_STATE > 128
    D_state_over = 160
    A_over = -torch.rand(D_sz, D_state_over, device="cuda", dtype=torch.float32)
    B_over = torch.randn(B_sz, D_state_over, L_sz, device="cuda", dtype=torch.float32) * 0.1
    C_over = torch.randn(B_sz, D_state_over, L_sz, device="cuda", dtype=torch.float32) * 0.1
    d_dummy = torch.empty(0, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="selective_scan_fwd: D_STATE=.* > 128 not supported by generic kernel"):
        mod.selective_scan_fwd(u.contiguous(), delta.contiguous(), A_over.contiguous(), B_over.contiguous(), C_over.contiguous(), d_dummy, False)

    out_fallback, _ = selective_scan(u, delta, A_over, B_over, C_over)
    ref_fallback, _ = selective_scan_pytorch_ref(u, delta, A_over, B_over, C_over)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


# 10. state_update
def test_state_update_silent_truncation():
    from neuralops.core.cuda_kernels.state_update.state_autograd import (
        _get_module,
        recurrent_linear_step,
        recurrent_linear_step_pytorch_ref,
    )
    mod = _get_module()
    B, H, Dv = 2, 4, 32
    v_t = torch.randn(B, H, Dv, device="cuda", dtype=torch.float32) * 0.1

    # Valid Dk <= 128
    Dk_valid = 32
    q_t = (torch.randn(B, H, Dk_valid, device="cuda", dtype=torch.float32) * 0.1).abs() + 0.1
    k_t = (torch.randn(B, H, Dk_valid, device="cuda", dtype=torch.float32) * 0.1).abs() + 0.1
    kv = torch.randn(B, H, Dk_valid, Dv, device="cuda", dtype=torch.float32) * 0.1
    z = (torch.randn(B, H, Dk_valid, device="cuda", dtype=torch.float32) * 0.1).abs() + 0.1

    out_valid, kv_valid, z_valid = recurrent_linear_step(q_t, k_t, v_t, kv, z)
    ref_valid, kv_ref, z_ref = recurrent_linear_step_pytorch_ref(q_t, k_t, v_t, kv, z)
    assert (out_valid - ref_valid).abs().max() < 1e-3

    # Oversized Dk > 128
    Dk_over = 160
    q_over = (torch.randn(B, H, Dk_over, device="cuda", dtype=torch.float32) * 0.1).abs() + 0.1
    k_over = (torch.randn(B, H, Dk_over, device="cuda", dtype=torch.float32) * 0.1).abs() + 0.1
    kv_over = torch.randn(B, H, Dk_over, Dv, device="cuda", dtype=torch.float32) * 0.1
    z_over = (torch.randn(B, H, Dk_over, device="cuda", dtype=torch.float32) * 0.1).abs() + 0.1

    with pytest.raises(RuntimeError, match="state_update_fwd: Dk=.* > 128 not supported"):
        mod.recurrent_linear_step_fwd(q_over.contiguous(), k_over.contiguous(), v_t.contiguous(), kv_over.contiguous(), z_over.contiguous(), 1e-6)

    out_fallback, _, _ = recurrent_linear_step(q_over, k_over, v_t, kv_over, z_over)
    ref_fallback, _, _ = recurrent_linear_step_pytorch_ref(q_over, k_over, v_t, kv_over, z_over)
    assert (out_fallback - ref_fallback).abs().max() < 1e-3


# 11. tt_contraction
def test_tt_contraction_silent_truncation():
    from neuralops.core.cuda_kernels.tt_contraction.tt_autograd import (
        _get_module,
        tt_linear,
        tt_linear_pytorch_ref,
    )
    mod = _get_module()
    M, K, N = 4, 64, 64
    x = torch.randn(M, K, device="cuda", dtype=torch.float32) * 0.1

    # Valid R <= 64
    R_valid = 32
    core0 = torch.randn(1, 8, 8, R_valid, device="cuda", dtype=torch.float32) * 0.1
    core1 = torch.randn(R_valid, 8, 8, 1, device="cuda", dtype=torch.float32) * 0.1

    out_valid = tt_linear(x, core0, core1, (8, 8), (8, 8))
    ref_valid = tt_linear_pytorch_ref(x, core0, core1, (8, 8), (8, 8))
    assert (out_valid - ref_valid).abs().max() < 1e-3

    # Oversized R > 64
    R_over = 80
    core0_over = torch.randn(1, 8, 8, R_over, device="cuda", dtype=torch.float32) * 0.1
    core1_over = torch.randn(R_over, 8, 8, 1, device="cuda", dtype=torch.float32) * 0.1

    with pytest.raises(RuntimeError, match="tt_linear_fwd: R=.* > 64 not supported"):
        c0 = core0_over.squeeze(0).contiguous()
        c1 = core1_over.squeeze(-1).contiguous()
        mod.tt_linear_fwd(x.contiguous(), c0, c1, 8, 8, 8, 8, None)

    out_fallback = tt_linear(x, core0_over, core1_over, (8, 8), (8, 8))
    ref_fallback = tt_linear_pytorch_ref(x, core0_over, core1_over, (8, 8), (8, 8))
    assert (out_fallback - ref_fallback).abs().max() < 1e-3
