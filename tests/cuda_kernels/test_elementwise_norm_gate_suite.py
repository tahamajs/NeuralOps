#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Production Verification Test Suite for Agent 3 (Elementwise / Norm / Gating CUDA Kernels)
Verifies all 20 Target Kernels on NVIDIA Ada Lovelace SM 8.9:
FP32 max_diff < 1e-4, FP16 max_diff <= 0.05 (tolerance accounting for FP32 accumulator vs 16-bit intermediate quantization),
zero NaNs/Infs, contiguous & multi-shape layouts.
"""

import pytest
import torch
import neuralops.core.cuda_kernels as k

@pytest.fixture(autouse=True)
def setup_cuda():
    assert torch.cuda.is_available(), "CUDA GPU host required for test suite execution"
    torch.cuda.empty_cache()

def check_parity(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype, tol_fp32: float = 1e-4, tol_fp16: float = 0.05):
    assert not torch.isnan(actual).any(), "NaN detected in CUDA output!"
    assert not torch.isinf(actual).any(), "Inf detected in CUDA output!"
    diff = (actual.float() - expected.float()).abs().max().item()
    tol = tol_fp32 if dtype == torch.float32 else tol_fp16
    assert diff <= tol, f"Parity failure: diff={diff:.6e} > tol={tol}"

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_softplus(dtype):
    x = torch.randn(16, 512, device='cuda', dtype=dtype)
    from neuralops.core.cuda_kernels.softplus.softplus_autograd import _get_module as get_mod, fused_softplus_pytorch_ref
    out_c = get_mod().softplus_fwd(x, 1.0, 20.0)
    out_r = fused_softplus_pytorch_ref(x.float(), 1.0, 20.0).to(dtype)
    check_parity(out_c, out_r, dtype, tol_fp16=0.01)

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_channel_shuffle(dtype):
    x = torch.randn(8, 32, 128, device='cuda', dtype=dtype)
    from neuralops.core.cuda_kernels.channel_shuffle.shuffle_autograd import _get_module as get_mod, channel_shuffle_pytorch_ref
    out_c = get_mod().channel_shuffle_fwd(x, 4)
    out_r = channel_shuffle_pytorch_ref(x, 4)
    check_parity(out_c, out_r, dtype, tol_fp16=0.01)

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_entropy_gate(dtype):
    x = torch.randn(32, 32, device='cuda', dtype=dtype)
    V = torch.randn(16, 32, device='cuda', dtype=dtype)
    U = torch.randn(16, 64, device='cuda', dtype=dtype)
    from neuralops.core.cuda_kernels.entropy_gate.entropy_gate_autograd import _get_module as get_mod
    out_c = get_mod().spectral_entropy_svd_gate_fwd(x, V, U, 16, None, None)
    z = torch.matmul(x.float(), V.float().t())
    out_r = torch.matmul(z, U.float()).to(dtype)
    check_parity(out_c, out_r, dtype, tol_fp16=0.05)

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_bitonic_topk(dtype):
    x = torch.randn(32, 128, device='cuda', dtype=dtype)
    from neuralops.core.cuda_kernels.bitonic_topk.bitonic_topk_autograd import _get_module as get_mod, bitonic_topk_pytorch_ref
    out_c = get_mod().bitonic_topk_indices_fwd(x, 8)
    out_r = bitonic_topk_pytorch_ref(x, 8)
    val_c = torch.gather(x, -1, out_c)
    val_r = torch.gather(x, -1, out_r)
    check_parity(val_c, val_r, dtype, tol_fp16=0.01)

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_fused_dropout_add(dtype):
    x = torch.randn(16, 256, device='cuda', dtype=dtype)
    res = torch.randn(16, 256, device='cuda', dtype=dtype)
    from neuralops.core.cuda_kernels.fused_dropout_add.dropout_add_autograd import _get_module as get_mod, fused_dropout_add_pytorch_ref
    out_c = get_mod().fused_dropout_add_fwd(x, res, 0.0, False, 0)
    out_r = fused_dropout_add_pytorch_ref(x, res, 0.0, False)
    check_parity(out_c, out_r, dtype, tol_fp16=0.01)

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_aegis_gate_norm(dtype):
    b0 = torch.randn(32, 128, device='cuda', dtype=dtype)
    b1 = torch.randn(32, 128, device='cuda', dtype=dtype)
    w = torch.randn(32, 2, device='cuda', dtype=torch.float32)
    from neuralops.core.cuda_kernels.aegis_gate_norm.aegis_gate_norm_autograd import _get_module as get_mod, aegis_gate_norm_pytorch_ref
    out_c = get_mod().aegis_gate_norm_2branch_fwd(b0, b1, w, 1e-5)
    out_r = aegis_gate_norm_pytorch_ref(b0, b1, w, 1e-5)
    check_parity(out_c, out_r, dtype, tol_fp16=0.01)

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_rms_variance_match(dtype):
    y = torch.randn(32, 128, device='cuda', dtype=dtype)
    v = torch.randn(32, 128, device='cuda', dtype=dtype)
    from neuralops.core.cuda_kernels.rms_variance_match.rms_variance_match_autograd import _get_module as get_mod, rms_variance_match_pytorch_ref
    out_c = get_mod().rms_variance_match_fwd(y, v, 0.25, 4.0)
    out_r = rms_variance_match_pytorch_ref(y, v, 0.25, 4.0)
    check_parity(out_c, out_r, dtype, tol_fp16=0.01)

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_entmax_bisect(dtype):
    sc = torch.randn(16, 128, device='cuda', dtype=dtype)
    from neuralops.core.cuda_kernels.entmax_bisect.entmax_bisect_autograd import _get_module as get_mod, entmax_bisect
    out_c = get_mod().entmax15_fwd(sc, 15)
    out_r = entmax_bisect(sc, 15)
    check_parity(out_c, out_r, dtype, tol_fp16=0.01)

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_fwht_transform(dtype):
    x = torch.randn(16, 128, device='cuda', dtype=dtype)
    from neuralops.core.cuda_kernels.fwht_transform.fwht_transform_autograd import _get_module as get_mod, fwht_transform_pytorch_ref
    out_c = get_mod().fwht_fwd(x, 128)
    out_r = fwht_transform_pytorch_ref(x, 128)
    check_parity(out_c, out_r, dtype, tol_fp16=0.01)

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_unified_rope_offset(dtype):
    x = torch.randn(2, 4, 32, 64, device='cuda', dtype=dtype)
    cos = torch.randn(64, 32, device='cuda', dtype=torch.float32)
    sin = torch.randn(64, 32, device='cuda', dtype=torch.float32)
    from neuralops.core.cuda_kernels.unified_rope_offset.unified_rope_offset_autograd import _get_module as get_mod, unified_rope_offset_pytorch_ref
    out_c = get_mod().unified_rope_offset_fwd(x, cos, sin, 64, 0)
    out_r = unified_rope_offset_pytorch_ref(x, cos, sin, 64, 0)
    check_parity(out_c, out_r, dtype, tol_fp16=0.01)

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_spectra_v2(dtype):
    x = torch.randn(8, 32, 128, device='cuda', dtype=dtype)
    f = torch.randn(128, device='cuda', dtype=dtype)
    from neuralops.core.cuda_kernels.spectra_v2.spectra_v2_autograd import _get_module as get_mod, spectra_v2_ref
    out_c = get_mod().spectra_v2_fwd(x, f, 0.05)
    out_r = spectra_v2_ref(x, f, 0.05)
    check_parity(out_c, out_r, dtype, tol_fp16=0.01)

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_spectral_shrinkage(dtype):
    xr = torch.randn(8, 64, device='cuda', dtype=dtype)
    xi = torch.randn(8, 64, device='cuda', dtype=dtype)
    wr = torch.randn(8, 64, device='cuda', dtype=dtype)
    wi = torch.randn(8, 64, device='cuda', dtype=dtype)
    tau = torch.randn(8, 64, device='cuda', dtype=dtype)
    from neuralops.core.cuda_kernels.spectral_shrinkage.spectral_shrinkage_autograd import _get_module as get_mod
    out = get_mod().spectral_shrinkage_fwd(xr, xi, wr, wi, tau)
    yr_c, yi_c = out[0], out[1]
    y_hat = torch.complex(xr.float(), xi.float()) * torch.complex(wr.float(), wi.float())
    mag = torch.abs(y_hat).clamp_min(1e-8)
    sig_tau = torch.sigmoid(tau.float())
    scale = torch.relu(mag - sig_tau) / mag
    exp = y_hat * scale
    check_parity(yr_c, exp.real.to(dtype), dtype, tol_fp16=0.01)
    check_parity(yi_c, exp.imag.to(dtype), dtype, tol_fp16=0.01)

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_fused_norm_linear(dtype):
    x = torch.randn(16, 64, device='cuda', dtype=dtype)
    w = torch.randn(32, 64, device='cuda', dtype=dtype)
    from neuralops.core.cuda_kernels.fused_norm_linear.norm_linear_autograd import _get_module as get_mod
    out_c = get_mod().fused_norm_linear_fwd(x, w, torch.empty(0, device='cuda'), 1e-6)[0]
    x_mean = x.float().mean(-1, keepdim=True)
    x_var = x.float().var(-1, keepdim=True, unbiased=False)
    x_hat = (x.float() - x_mean) * torch.rsqrt(x_var + 1e-6)
    out_f32 = torch.matmul(x_hat, w.float().t()).to(dtype)
    check_parity(out_c, out_f32, dtype, tol_fp16=0.01)

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_causal_conv1d(dtype):
    x = torch.randn(2, 16, 64, device='cuda', dtype=dtype)
    w = torch.randn(16, 4, device='cuda', dtype=dtype)
    from neuralops.core.cuda_kernels.causal_conv1d.conv1d_autograd import _get_module as get_mod, causal_conv1d_pytorch_ref
    out_c = get_mod().causal_conv1d_fwd(x, w, None, False)
    out_r = causal_conv1d_pytorch_ref(x, w, None, False)
    check_parity(out_c, out_r, dtype, tol_fp16=0.01)

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_soa_attn(dtype):
    q = torch.randn(4, 16, 32, device='cuda', dtype=dtype)
    k_t = torch.randn(4, 16, 32, device='cuda', dtype=dtype)
    v = torch.randn(4, 16, 32, device='cuda', dtype=dtype)
    from neuralops.core.cuda_kernels.soa_attn.soa_attn_autograd import _get_module as get_mod
    out_c = get_mod().soa_attn_fwd(q, k_t, v, 0.5)
    qk_32 = q.float() * k_t.float()
    phi_32 = qk_32 + 0.5 * (qk_32 * qk_32)
    out_r = (phi_32 * v.float()).to(dtype)
    check_parity(out_c, out_r, dtype, tol_fp16=0.01)

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_recurrent_hgrn2_step(dtype):
    x = torch.randn(8, 32, device='cuda', dtype=dtype)
    g = torch.randn(8, 32, device='cuda', dtype=dtype)
    a = torch.randn(8, 32, device='cuda', dtype=dtype)
    h = torch.randn(8, 32, device='cuda', dtype=dtype)
    from neuralops.core.cuda_kernels.recurrent_hgrn2_step.hgrn2_step_autograd import _get_module as get_mod
    out_c, h_c = get_mod().recurrent_hgrn2_step_fwd(x, g, a, h.clone())
    h_new32 = a.float() * h.float() + (1.0 - a.float()) * x.float()
    out_32 = (h_new32 * g.float()).to(dtype)
    check_parity(out_c, out_32, dtype, tol_fp16=0.01)
    check_parity(h_c, h_new32.to(dtype), dtype, tol_fp16=0.01)

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_state_update(dtype):
    q = torch.randn(2, 2, 16, device='cuda', dtype=dtype)
    k_t = torch.randn(2, 2, 16, device='cuda', dtype=dtype)
    v = torch.randn(2, 2, 16, device='cuda', dtype=dtype)
    kv = torch.randn(2, 2, 16, 16, device='cuda', dtype=dtype)
    z = torch.randn(2, 2, 16, device='cuda', dtype=dtype)
    from neuralops.core.cuda_kernels.state_update.state_autograd import _get_module as get_mod
    out_c, kvc, zc = get_mod().recurrent_linear_step_fwd(q, k_t, v, kv.clone(), z.clone(), 1e-6)
    z_new = z.float() + k_t.float()
    denom = (q.float() * z_new).sum(dim=-1, keepdim=True) + 1e-6
    kv_new = kv.float() + torch.einsum('bhk,bhv->bhkv', k_t.float(), v.float())
    num = torch.einsum('bhk,bhkv->bhv', q.float(), kv_new)
    out_exp = (num / denom).to(dtype)
    check_parity(out_c, out_exp, dtype, tol_fp16=0.01)
    check_parity(kvc, kv_new.to(dtype), dtype, tol_fp16=0.01)
    check_parity(zc, z_new.to(dtype), dtype, tol_fp16=0.01)

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_act_scale_gemm(dtype):
    x = torch.randn(16, 32, device='cuda', dtype=dtype)
    s = torch.randn(32, device='cuda', dtype=dtype)
    w = torch.randn(16, 32, device='cuda', dtype=dtype)
    from neuralops.core.cuda_kernels.act_scale_gemm.act_scale_autograd import _get_module as get_mod
    out_c = get_mod().act_scale_quant_gemm_fwd(x, s, w, None)
    out_exp = torch.matmul((x.float() * s.float().unsqueeze(0)), w.float().t()).to(dtype)
    check_parity(out_c, out_exp, dtype, tol_fp16=0.01)

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_kronecker_gemm(dtype):
    x = torch.randn(8, 32, device='cuda', dtype=dtype)
    A = torch.randn(4, 4, device='cuda', dtype=dtype)
    B = torch.randn(8, 8, device='cuda', dtype=dtype)
    from neuralops.core.cuda_kernels.kronecker_gemm.kronecker_autograd import _get_module as get_mod
    out_c = get_mod().kronecker_gemm_fwd(x, A, B, torch.empty(0, device='cuda'))
    T = torch.matmul(x.float().view(8, 4, 8), B.float().t())
    Y = torch.matmul(A.float(), T).view(8, 32).to(dtype)
    check_parity(out_c, Y, dtype, tol_fp16=0.01)

@pytest.mark.parametrize("dtype", [torch.float32, torch.half])
def test_fp4_gemm(dtype):
    x = torch.randn(16, 32, device='cuda', dtype=dtype)
    wp = torch.randint(0, 255, (16, 16), dtype=torch.uint8, device='cuda')
    scales = torch.randn(16, device='cuda', dtype=torch.float32)
    from neuralops.core.cuda_kernels.fp4_gemm.fp4_autograd import _get_module as get_mod, fp4_gemm_pytorch_ref
    out_c = get_mod().fp4_gemm_fwd(x, wp, scales, None)
    out_32 = fp4_gemm_pytorch_ref(x.float(), wp, scales, None).to(dtype)
    check_parity(out_c, out_32, dtype, tol_fp16=0.01)
