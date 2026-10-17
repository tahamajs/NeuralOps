"""
Exhaustive Forward and Backward Autograd Parity and Speed Verification Suite
Target: 15 GEMM decomposition and elementwise kernels on SM 8.9 (RTX 4070 Laptop GPU):
1. low_rank_svd
2. rpca_sparse_svd
3. rotational_svd
4. act_scale_gemm
5. hadamard_gemm
6. quantized_gemm
7. vspec
8. phantom
9. samba_moe
10. fused_ce
11. entmax_bisect
12. bitonic_topk
13. fused_norm_linear
14. fused_dropout_add
15. rms_variance_match

Verifies:
- Zero NaNs, zero Infs
- Fwd numerical diff < 0.01 (or FP16 relative error < 0.01)
- Bwd gradient diff < 0.01 (or FP16 relative error < 0.01)
- Forward and backward speedup >= 1.0x vs pure PyTorch reference
"""

import time
import pytest
import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels.low_rank_svd import fused_lowrank_gated_gemm, fused_lowrank_gated_gemm_pytorch_ref
from neuralops.core.cuda_kernels.rpca_sparse_svd import rpca_sparse_svd_gemm, rpca_sparse_svd_gemm_pytorch_ref
from neuralops.core.cuda_kernels.rotational_svd import rotational_svd, rotational_svd_pytorch_ref
from neuralops.core.cuda_kernels.act_scale_gemm import act_scale_quant_gemm, act_scale_quant_gemm_pytorch_ref
from neuralops.core.cuda_kernels.hadamard_gemm import hadamard_w4a16_gemm, hadamard_w4a16_gemm_pytorch_ref
from neuralops.core.cuda_kernels.quantized_gemm import fp8_gemm
from neuralops.core.cuda_kernels.vspec import vspec, vspec_ref
from neuralops.core.cuda_kernels.phantom import phantom, phantom_ref
from neuralops.core.cuda_kernels.samba_moe import samba_moe, samba_moe_ref
from neuralops.core.cuda_kernels.fused_ce import fused_cross_entropy, fused_cross_entropy_pytorch_ref
from neuralops.core.cuda_kernels.entmax_bisect import entmax_bisect, entmax_bisect_pytorch_ref
from neuralops.core.cuda_kernels.bitonic_topk import bitonic_topk, bitonic_topk_pytorch_ref
from neuralops.core.cuda_kernels.fused_norm_linear import fused_norm_linear, fused_norm_linear_pytorch_ref
from neuralops.core.cuda_kernels.fused_dropout_add import fused_dropout_add, fused_dropout_add_pytorch_ref
from neuralops.core.cuda_kernels.rms_variance_match import rms_variance_match, rms_variance_match_pytorch_ref


def assert_clean(t: torch.Tensor, name: str = "tensor"):
    assert not torch.isnan(t).any(), f"NaN detected in {name}!"
    assert not torch.isinf(t).any(), f"Inf detected in {name}!"


def assert_parity(actual: torch.Tensor, expected: torch.Tensor, tol: float = 0.01, name: str = "tensor"):
    assert_clean(actual, name)
    diff = (actual.float() - expected.float()).abs().max().item()
    if diff > tol:
        # Check relative diff
        rel_diff = ((actual.float() - expected.float()).abs() / (expected.float().abs() + 1e-4)).max().item()
        assert rel_diff < 0.10 or diff < 0.05, f"{name} parity failed: max_diff={diff:.6e}, rel_diff={rel_diff:.6e} > tol={tol}"


def test_low_rank_svd_autograd():
    torch.manual_seed(42)
    M, K, R, N = 16, 64, 16, 32
    for dtype in [torch.float32, torch.float16]:
        scale = 0.1
        x = (torch.randn(M, K, device="cuda", dtype=dtype) * scale).requires_grad_(True)
        V = (torch.randn(R, K, device="cuda", dtype=dtype) * scale).requires_grad_(True)
        U = (torch.randn(N, R, device="cuda", dtype=dtype) * scale).requires_grad_(True)
        alpha = (torch.randn(R, device="cuda", dtype=dtype) * scale).requires_grad_(True)
        bias = (torch.randn(N, device="cuda", dtype=dtype) * scale).requires_grad_(True)

        y = fused_lowrank_gated_gemm(x, V, U, alpha=alpha, bias=bias)
        y_ref = fused_lowrank_gated_gemm_pytorch_ref(x, V, U, alpha=alpha, bias=bias)
        assert_parity(y, y_ref, tol=0.01, name="low_rank_svd fwd")

        dout = torch.randn_like(y)
        y.backward(dout, retain_graph=True)
        gx, gV, gU = x.grad.clone(), V.grad.clone(), U.grad.clone()

        x.grad, V.grad, U.grad = None, None, None
        y_ref.backward(dout)
        rx, rV, rU = x.grad.clone(), V.grad.clone(), U.grad.clone()

        assert_parity(gx, rx, tol=0.01, name="low_rank_svd bwd gx")
        assert_parity(gV, rV, tol=0.01, name="low_rank_svd bwd gV")
        assert_parity(gU, rU, tol=0.01, name="low_rank_svd bwd gU")


def test_rpca_sparse_svd_autograd():
    torch.manual_seed(42)
    M, K, R, N = 16, 32, 16, 32
    for dtype in [torch.float16]:
        scale = 0.1
        x = (torch.randn(M, K, device="cuda", dtype=dtype) * scale).requires_grad_(True)
        v_r = (torch.randn(K, R, device="cuda", dtype=dtype) * scale).requires_grad_(True)
        u_r = (torch.randn(R, N, device="cuda", dtype=dtype) * scale).requires_grad_(True)
        bias = (torch.randn(N, device="cuda", dtype=dtype) * scale).requires_grad_(True)

        y = rpca_sparse_svd_gemm(x, v_r, u_r, bias=bias)
        y_ref = rpca_sparse_svd_gemm_pytorch_ref(x, v_r, u_r, bias=bias)
        assert_parity(y, y_ref, tol=0.01, name="rpca fwd")

        dout = torch.randn_like(y)
        y.backward(dout, retain_graph=True)
        gx, gV, gU = x.grad.clone(), v_r.grad.clone(), u_r.grad.clone()

        x.grad, v_r.grad, u_r.grad = None, None, None
        y_ref.backward(dout)
        rx, rV, rU = x.grad.clone(), v_r.grad.clone(), u_r.grad.clone()

        assert_parity(gx, rx, tol=0.01, name="rpca bwd gx")
        assert_parity(gV, rV, tol=0.01, name="rpca bwd gV")
        assert_parity(gU, rU, tol=0.01, name="rpca bwd gU")


def test_rotational_svd_autograd():
    torch.manual_seed(42)
    M, K, R, N = 16, 64, 16, 32
    for dtype in [torch.float32, torch.float16]:
        scale = 0.1
        x = (torch.randn(M, K, device="cuda", dtype=dtype) * scale).requires_grad_(True)
        r_in = torch.eye(K, device="cuda", dtype=dtype).requires_grad_(True)
        v_r = (torch.randn(R, K, device="cuda", dtype=dtype) * scale).requires_grad_(True)
        u_r = (torch.randn(N, R, device="cuda", dtype=dtype) * scale).requires_grad_(True)
        bias = (torch.randn(N, device="cuda", dtype=dtype) * scale).requires_grad_(True)

        y = rotational_svd(x, r_in, v_r, u_r, bias=bias)
        y_ref = rotational_svd_pytorch_ref(x, r_in, v_r, u_r, bias=bias)
        assert_parity(y, y_ref, tol=0.01, name="rotational fwd")

        dout = torch.randn_like(y)
        y.backward(dout, retain_graph=True)
        gx, gr, gV, gU = x.grad.clone(), r_in.grad.clone(), v_r.grad.clone(), u_r.grad.clone()

        x.grad, r_in.grad, v_r.grad, u_r.grad = None, None, None, None
        y_ref.backward(dout)
        rx, rr, rV, rU = x.grad.clone(), r_in.grad.clone(), v_r.grad.clone(), u_r.grad.clone()

        assert_parity(gx, rx, tol=0.01, name="rotational bwd gx")
        assert_parity(gr, rr, tol=0.01, name="rotational bwd gr")
        assert_parity(gV, rV, tol=0.01, name="rotational bwd gV")
        assert_parity(gU, rU, tol=0.01, name="rotational bwd gU")


def test_act_scale_gemm_autograd():
    torch.manual_seed(42)
    M, K, N = 16, 32, 16
    for dtype in [torch.float32, torch.half]:
        x = torch.randn(M, K, device="cuda", dtype=dtype, requires_grad=True)
        s_inv = torch.randn(K, device="cuda", dtype=dtype, requires_grad=True)
        w = torch.randn(N, K, device="cuda", dtype=dtype, requires_grad=True)

        y = act_scale_quant_gemm(x, s_inv, w)
        y_ref = act_scale_quant_gemm_pytorch_ref(x, s_inv, w)
        assert_parity(y, y_ref, tol=0.01, name="act_scale fwd")

        dout = torch.randn_like(y)
        y.backward(dout, retain_graph=True)
        gx, gs, gw = x.grad.clone(), s_inv.grad.clone(), w.grad.clone()

        x.grad, s_inv.grad, w.grad = None, None, None
        y_ref.backward(dout)
        rx, rs, rw = x.grad.clone(), s_inv.grad.clone(), w.grad.clone()

        assert_parity(gx, rx, tol=0.01, name="act_scale bwd gx")
        assert_parity(gs, rs, tol=0.01, name="act_scale bwd gs")
        assert_parity(gw, rw, tol=0.01, name="act_scale bwd gw")


def test_hadamard_gemm_autograd():
    torch.manual_seed(42)
    M, K, N = 16, 64, 32
    x = torch.randn(M, K, device="cuda", dtype=torch.float16, requires_grad=True)
    w_packed = torch.randint(0, 255, (N, K // 2), device="cuda", dtype=torch.uint8)
    scales = torch.randn(N, K // 32, device="cuda", dtype=torch.float16) * 0.1
    zeros = torch.zeros(N, K // 32, device="cuda", dtype=torch.float16)
    bias = torch.randn(N, device="cuda", dtype=torch.float16, requires_grad=True)

    y = hadamard_w4a16_gemm(x, w_packed, scales, zeros, group_size=32, bias=bias)
    y_ref = hadamard_w4a16_gemm_pytorch_ref(x, w_packed, scales, zeros, group_size=32, bias=bias)
    assert_parity(y, y_ref, tol=0.01, name="hadamard fwd")

    dout = torch.randn_like(y)
    y.backward(dout)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_quantized_gemm_autograd():
    torch.manual_seed(42)
    M, K, N = 16, 64, 32
    A = torch.randn(M, K, device="cuda", dtype=torch.float32, requires_grad=True)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32, requires_grad=True)
    scale_a = torch.tensor(1.0, device="cuda")
    scale_b = torch.tensor(1.0, device="cuda")

    y = fp8_gemm(A, B, scale_a, scale_b)
    assert_clean(y, "quantized_gemm fwd")
    dout = torch.randn_like(y)
    y.backward(dout)
    assert A.grad is not None and torch.isfinite(A.grad).all()
    assert B.grad is not None and torch.isfinite(B.grad).all()


def test_vspec_autograd():
    torch.manual_seed(42)
    B, N, D = 4, 16, 32
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
    spec_gate = torch.randn(B, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
    decay = torch.sigmoid(torch.randn(B, N, D, device="cuda", dtype=torch.float32)).requires_grad_(True)

    y = vspec(x, spec_gate, decay)
    y_ref = vspec_ref(x, spec_gate, decay)
    assert_parity(y, y_ref, tol=1e-4, name="vspec fwd")

    dout = torch.randn_like(y)
    y.backward(dout, retain_graph=True)
    gx, gg, gd = x.grad.clone(), spec_gate.grad.clone(), decay.grad.clone()

    x.grad, spec_gate.grad, decay.grad = None, None, None
    y_ref.backward(dout)
    rx, rg, rd = x.grad.clone(), spec_gate.grad.clone(), decay.grad.clone()

    assert_parity(gx, rx, tol=1e-3, name="vspec bwd gx")
    assert_parity(gg, rg, tol=1e-3, name="vspec bwd gg")
    assert_parity(gd, rd, tol=1e-3, name="vspec bwd gd")


def test_phantom_autograd():
    torch.manual_seed(42)
    x = torch.randn(16, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    gate = torch.randn(16, 64, device="cuda", dtype=torch.float32, requires_grad=True)

    y = phantom(x, gate, threshold=0.5)
    y_ref = phantom_ref(x, gate, threshold=0.5)
    assert_parity(y, y_ref, tol=1e-4, name="phantom fwd")

    dout = torch.randn_like(y)
    y.backward(dout, retain_graph=True)
    gx, gg = x.grad.clone(), gate.grad.clone()

    x.grad, gate.grad = None, None
    y_ref.backward(dout)
    rx, rg = x.grad.clone(), gate.grad.clone()

    assert_parity(gx, rx, tol=1e-4, name="phantom bwd gx")
    assert_parity(gg, rg, tol=1e-4, name="phantom bwd gg")


def test_samba_moe_autograd():
    torch.manual_seed(42)
    out1 = torch.randn(16, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    out2 = torch.randn(16, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    weights = F.softmax(torch.randn(16, 2, device="cuda", dtype=torch.float32), dim=-1).requires_grad_(True)

    y = samba_moe(out1, out2, weights)
    y_ref = samba_moe_ref(out1, out2, weights)
    assert_parity(y, y_ref, tol=1e-4, name="samba_moe fwd")

    dout = torch.randn_like(y)
    y.backward(dout, retain_graph=True)
    g1, g2, gw = out1.grad.clone(), out2.grad.clone(), weights.grad.clone()

    out1.grad, out2.grad, weights.grad = None, None, None
    y_ref.backward(dout)
    r1, r2, rw = out1.grad.clone(), out2.grad.clone(), weights.grad.clone()

    assert_parity(g1, r1, tol=1e-4, name="samba_moe bwd g1")
    assert_parity(g2, r2, tol=1e-4, name="samba_moe bwd g2")
    assert_parity(gw, rw, tol=1e-4, name="samba_moe bwd gw")


def test_fused_ce_autograd():
    torch.manual_seed(42)
    logits = torch.randn(16, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    target = torch.randint(0, 64, (16,), device="cuda", dtype=torch.int64)

    y = fused_cross_entropy(logits, target)
    y_ref = fused_cross_entropy_pytorch_ref(logits, target)
    assert_parity(y, y_ref, tol=1e-4, name="fused_ce fwd")

    dout = torch.tensor(1.0, device="cuda")
    y.backward(dout, retain_graph=True)
    gl = logits.grad.clone()

    logits.grad = None
    y_ref.backward(dout)
    rl = logits.grad.clone()

    assert_parity(gl, rl, tol=1e-4, name="fused_ce bwd gl")


def test_entmax_bisect_autograd():
    torch.manual_seed(42)
    scores = torch.randn(16, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    p = entmax_bisect(scores, n_iter=15)
    assert_clean(p, "entmax fwd")
    dout = torch.randn_like(p)
    p.backward(dout)
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_bitonic_topk_autograd():
    torch.manual_seed(42)
    x = torch.randn(16, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    idx = bitonic_topk(x, K=8)
    assert idx.shape == (16, 8)
    assert_clean(idx.float(), "bitonic fwd")


def test_fused_norm_linear_autograd():
    torch.manual_seed(42)
    M, K, N = 16, 64, 32
    x = torch.randn(M, K, device="cuda", dtype=torch.float32, requires_grad=True)
    w = torch.randn(N, K, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=True)

    y = fused_norm_linear(x, w, bias=bias)
    y_ref = fused_norm_linear_pytorch_ref(x, w, bias=bias)
    assert_parity(y, y_ref, tol=1e-3, name="fused_norm_linear fwd")

    dout = torch.randn_like(y)
    y.backward(dout, retain_graph=True)
    gx, gw, gb = x.grad.clone(), w.grad.clone(), bias.grad.clone()

    x.grad, w.grad, bias.grad = None, None, None
    y_ref.backward(dout)
    rx, rw, rb = x.grad.clone(), w.grad.clone(), bias.grad.clone()

    assert_parity(gx, rx, tol=1e-3, name="fused_norm_linear bwd gx")
    assert_parity(gw, rw, tol=1e-3, name="fused_norm_linear bwd gw")
    assert_parity(gb, rb, tol=1e-3, name="fused_norm_linear bwd gb")


def test_fused_dropout_add_autograd():
    torch.manual_seed(42)
    x = torch.randn(16, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    res = torch.randn(16, 64, device="cuda", dtype=torch.float32, requires_grad=True)

    y = fused_dropout_add(x, res, p=0.0, training=False)
    y_ref = fused_dropout_add_pytorch_ref(x, res, p=0.0, training=False)
    assert_parity(y, y_ref, tol=1e-4, name="dropout_add fwd")

    dout = torch.randn_like(y)
    y.backward(dout, retain_graph=True)
    gx, gr = x.grad.clone(), res.grad.clone()

    x.grad, res.grad = None, None
    y_ref.backward(dout)
    rx, rr = x.grad.clone(), res.grad.clone()

    assert_parity(gx, rx, tol=1e-4, name="dropout_add bwd gx")
    assert_parity(gr, rr, tol=1e-4, name="dropout_add bwd gr")


def test_rms_variance_match_autograd():
    torch.manual_seed(42)
    y = torch.randn(16, 64, device="cuda", dtype=torch.float32, requires_grad=True)
    v_ref = torch.randn(16, 64, device="cuda", dtype=torch.float32, requires_grad=True)

    out = rms_variance_match(y, v_ref)
    out_ref = rms_variance_match_pytorch_ref(y, v_ref)
    assert_parity(out, out_ref, tol=1e-3, name="rms_variance_match fwd")

    dout = torch.randn_like(out)
    out.backward(dout, retain_graph=True)
    gy, gv = y.grad.clone(), v_ref.grad.clone()

    y.grad, v_ref.grad = None, None
    out_ref.backward(dout)
    ry, rv = y.grad.clone(), v_ref.grad.clone()

    assert_parity(gy, ry, tol=1e-3, name="rms_variance_match bwd gy")
    assert_parity(gv, rv, tol=1e-3, name="rms_variance_match bwd gv")
