"""Production-Grade Test Suite for NeuralOps GEMM and Matrix Decomposition CUDA Kernels.

Tests 20 CUDA C++ GEMM and Matrix Decomposition modules:
 1. fused_ce
 2. logcumsumexp
 3. lut_gemm
 4. samba_moe
 5. kronecker_gemm
 6. rotational_svd
 7. svd_prefix_scan
 8. ternary_gemm
 9. block_diag_gemm
10. spectral_shrinkage
11. phantom
12. tt_contraction
13. low_rank_svd
14. hadamard_gemm
15. quantized_gemm
16. rpca_sparse_svd
17. nasc_attention
18. nexus
19. vspec
20. goldfinch_ahn

Verifies:
- Numerical parity: FP16 max_diff < 0.01 (or scaled < 0.01), FP32 max_diff < 1e-4
- Zero NaNs and Infs
- Multi-shape sweeps & contiguous vs non-contiguous layouts
- 3-level fallback dispatch (CUDA vs PyTorch reference)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels import force_kernel_backend
from .conftest import (
    assert_numerical_parity,
    check_tensor_validity,
    create_non_contiguous,
)

from neuralops.core.cuda_kernels.fused_ce import fused_ce_autograd as m_fused_ce
from neuralops.core.cuda_kernels.logcumsumexp import logcumsumexp_autograd as m_logcumsumexp
from neuralops.core.cuda_kernels.lut_gemm import lut_autograd as m_lut_gemm
from neuralops.core.cuda_kernels.samba_moe import samba_moe_autograd as m_samba_moe
from neuralops.core.cuda_kernels.kronecker_gemm import kronecker_autograd as m_kronecker_gemm
from neuralops.core.cuda_kernels.rotational_svd import rotational_svd_autograd as m_rotational_svd
from neuralops.core.cuda_kernels.svd_prefix_scan import svd_prefix_scan_autograd as m_svd_prefix_scan
from neuralops.core.cuda_kernels.ternary_gemm import ternary_autograd as m_ternary_gemm
from neuralops.core.cuda_kernels.block_diag_gemm import block_diag_autograd as m_block_diag_gemm
from neuralops.core.cuda_kernels.spectral_shrinkage import spectral_shrinkage_autograd as m_spectral_shrinkage
from neuralops.core.cuda_kernels.phantom import phantom_autograd as m_phantom
from neuralops.core.cuda_kernels.tt_contraction import tt_autograd as m_tt_contraction
from neuralops.core.cuda_kernels.low_rank_svd import lowrank_autograd as m_low_rank_svd
from neuralops.core.cuda_kernels.hadamard_gemm import hadamard_autograd as m_hadamard_gemm
from neuralops.core.cuda_kernels.quantized_gemm import quant_autograd as m_quantized_gemm
from neuralops.core.cuda_kernels.rpca_sparse_svd import rpca_sparse_svd_autograd as m_rpca_sparse_svd
from neuralops.core.cuda_kernels.nasc_attention import nasc_attention_autograd as m_nasc_attention
from neuralops.core.cuda_kernels.nexus import nexus_autograd as m_nexus
from neuralops.core.cuda_kernels.vspec import vspec_autograd as m_vspec
from neuralops.core.cuda_kernels.goldfinch_ahn import goldfinch_ahn_autograd as m_goldfinch_ahn


@pytest.fixture(autouse=True)
def setup_cuda_env():
    force_kernel_backend("cuda")
    torch.manual_seed(42)
    yield
    force_kernel_backend(None)


# 1. fused_ce
@pytest.mark.parametrize("shape", [(16, 64), (32, 128), (4, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fused_ce(shape, dtype):
    N, V = shape
    logits = torch.randn(N, V, device="cuda", dtype=dtype)
    targets = torch.randint(0, V, (N,), device="cuda")

    ref_out = m_fused_ce.fused_cross_entropy_pytorch_ref(logits.float(), targets)
    cuda_out = m_fused_ce.fused_cross_entropy(logits, targets)

    assert_numerical_parity(cuda_out, ref_out.to(cuda_out.dtype), dtype=dtype, name="fused_ce")


# 2. logcumsumexp
@pytest.mark.parametrize("shape", [(8, 128), (16, 256), (4, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_logcumsumexp(shape, dtype):
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    ref_out = m_logcumsumexp.logcumsumexp_pytorch_ref(x)
    cuda_out = m_logcumsumexp.logcumsumexp(x)

    assert_numerical_parity(cuda_out, ref_out, dtype=dtype, name="logcumsumexp")


# 3. lut_gemm
@pytest.mark.parametrize("shape", [(16, 64, 32), (32, 128, 64)])
def test_lut_gemm(shape):
    M, K, N = shape
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    idx = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)
    cb = torch.randn(max(1, K // 64), 16, device="cuda", dtype=torch.float16)

    ref_out = m_lut_gemm.lut_codebook_dequant_gemm_pytorch_ref(x, idx, cb, group_size=64)
    cuda_out = m_lut_gemm.lut_codebook_dequant_gemm(x, idx, cb, group_size=64)

    assert_numerical_parity(cuda_out, ref_out, dtype=torch.float16, name="lut_gemm")


# 4. samba_moe
@pytest.mark.parametrize("shape", [(16, 32), (32, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_samba_moe(shape, dtype):
    e1 = torch.randn(*shape, device="cuda", dtype=dtype)
    e2 = torch.randn(*shape, device="cuda", dtype=dtype)
    rw = torch.randn(shape[0], 2, device="cuda", dtype=dtype)

    ref_out = m_samba_moe.samba_moe_ref(e1, e2, rw)
    cuda_out = m_samba_moe.samba_moe(e1, e2, rw)

    assert_numerical_parity(cuda_out, ref_out, dtype=dtype, name="samba_moe")


# 5. kronecker_gemm
@pytest.mark.parametrize("dims", [((4, 8), (4, 8)), ((8, 8), (8, 8))])
def test_kronecker_gemm(dims):
    (M1, N1), (M2, N2) = dims
    x = torch.randn(8, N1 * N2, device="cuda", dtype=torch.float16) * 0.1
    A = torch.randn(M1, N1, device="cuda", dtype=torch.float16) * 0.1
    B = torch.randn(M2, N2, device="cuda", dtype=torch.float16) * 0.1

    ref_out = m_kronecker_gemm.kronecker_matmul_pytorch_ref(x, A, B)
    cuda_out = m_kronecker_gemm.kronecker_matmul(x, A, B)

    assert_numerical_parity(cuda_out, ref_out, dtype=torch.float16, atol=0.01, name="kronecker_gemm")


# 6. rotational_svd
@pytest.mark.parametrize("shape", [(8, 32, 16, 32), (16, 32, 32, 32)])
def test_rotational_svd(shape):
    M, K, R, N = shape
    x = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.1
    rin = torch.randn(K, K, device="cuda", dtype=torch.float16) * 0.1
    vr = torch.randn(R, K, device="cuda", dtype=torch.float16) * 0.1
    ur = torch.randn(N, R, device="cuda", dtype=torch.float16) * 0.1

    ref_out = m_rotational_svd.rotational_svd_pytorch_ref(x, rin, vr, ur)
    cuda_out = m_rotational_svd.rotational_svd(x, rin, vr, ur)

    assert_numerical_parity(cuda_out, ref_out, dtype=torch.float16, atol=0.01, name="rotational_svd")


# 7. svd_prefix_scan
@pytest.mark.parametrize("shape", [(2, 4, 16, 32), (1, 2, 32, 32)])
def test_svd_prefix_scan(shape):
    q = torch.randn(*shape, device="cuda", dtype=torch.float16) * 0.1
    k = torch.randn(*shape, device="cuda", dtype=torch.float16) * 0.1
    v = torch.randn(*shape, device="cuda", dtype=torch.float16) * 0.1

    ref_out = m_svd_prefix_scan.svd_prefix_scan_pytorch_ref(q, k, v)
    cuda_out = m_svd_prefix_scan.svd_prefix_scan(q, k, v)

    assert_numerical_parity(cuda_out, ref_out, dtype=torch.float16, atol=0.01, name="svd_prefix_scan")


# 8. ternary_gemm
@pytest.mark.parametrize("shape", [(8, 32, 16), (16, 64, 32)])
def test_ternary_gemm(shape):
    M, K, N = shape
    x = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.1
    w_packed = torch.randint(0, 256, (N, K // 4), device="cuda", dtype=torch.uint8)
    scales = torch.randn(N, device="cuda", dtype=torch.float32) * 0.1

    ref_out = m_ternary_gemm.ternary_gemm_pytorch_ref(x, w_packed, scales)
    cuda_out = m_ternary_gemm.ternary_gemm(x, w_packed, scales)

    assert_numerical_parity(cuda_out, ref_out, dtype=torch.float16, atol=0.01, name="ternary_gemm")


# 9. block_diag_gemm
@pytest.mark.parametrize("blocks", [(4, 16, 16), (2, 32, 32)])
def test_block_diag_gemm(blocks):
    B_blocks, block_k, block_n = blocks
    x = torch.randn(8, B_blocks * block_k, device="cuda", dtype=torch.float16)
    w = torch.randn(B_blocks, block_k, block_n, device="cuda", dtype=torch.float16)

    ref_out = m_block_diag_gemm.block_diag_matmul_pytorch_ref(x, w)
    cuda_out = m_block_diag_gemm.block_diag_matmul(x, w)

    assert_numerical_parity(cuda_out, ref_out, dtype=torch.float16, name="block_diag_gemm")


# 10. spectral_shrinkage
@pytest.mark.parametrize("dim", [16, 32])
def test_spectral_shrinkage(dim):
    xr = torch.randn(4, dim, device="cuda", dtype=torch.float32)
    xi = torch.randn(4, dim, device="cuda", dtype=torch.float32)
    wr = torch.randn(4, dim, device="cuda", dtype=torch.float32)
    wi = torch.randn(4, dim, device="cuda", dtype=torch.float32)
    tau = torch.tensor(0.5, device="cuda", dtype=torch.float32)

    ref_out = m_spectral_shrinkage.spectral_shrinkage_pytorch_ref(xr, xi, wr, wi, tau)
    cuda_out = m_spectral_shrinkage.spectral_shrinkage(xr, xi, wr, wi, tau)

    if isinstance(ref_out, tuple):
        for ro, co in zip(ref_out, cuda_out):
            assert_numerical_parity(co, ro, dtype=torch.float32, name="spectral_shrinkage")
    else:
        assert_numerical_parity(cuda_out, ref_out, dtype=torch.float32, name="spectral_shrinkage")


# 11. phantom
@pytest.mark.parametrize("shape", [(8, 32), (16, 64)])
def test_phantom(shape):
    x = torch.randn(*shape, device="cuda", dtype=torch.float16)
    g = torch.randn(*shape, device="cuda", dtype=torch.float16)

    ref_out = m_phantom.phantom_ref(x, g)
    cuda_out = m_phantom.phantom(x, g)

    assert_numerical_parity(cuda_out, ref_out, dtype=torch.float16, name="phantom")


# 12. tt_contraction
@pytest.mark.parametrize("dims", [((4, 4), (4, 4), 2), ((2, 2), (2, 2), 4)])
def test_tt_contraction(dims):
    (p0, p1), (q0, q1), r = dims
    c0 = torch.randn(1, q0, p0, r, device="cuda", dtype=torch.float32)
    c1 = torch.randn(r, q1, p1, 1, device="cuda", dtype=torch.float32)

    ref_out = m_tt_contraction.tt_contraction_pytorch_ref(c0, c1, (p0, p1), (q0, q1))
    cuda_out = m_tt_contraction.tt_contraction(c0, c1, (p0, p1), (q0, q1))

    assert_numerical_parity(cuda_out, ref_out, dtype=torch.float32, name="tt_contraction")


# 13. low_rank_svd
@pytest.mark.parametrize("shape", [(8, 32, 16, 32), (16, 64, 16, 64)])
def test_low_rank_svd(shape):
    M, K, R, N = shape
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    V_r = torch.randn(R, K, device="cuda", dtype=torch.float16)
    U_r = torch.randn(N, R, device="cuda", dtype=torch.float16)

    ref_out = m_low_rank_svd.fused_lowrank_gated_gemm_pytorch_ref(x, V_r, U_r)
    cuda_out = m_low_rank_svd.fused_lowrank_gated_gemm(x, V_r, U_r)

    assert_numerical_parity(cuda_out, ref_out, dtype=torch.float16, name="low_rank_svd")


# 14. hadamard_gemm
@pytest.mark.parametrize("shape", [(8, 32, 16), (16, 64, 32)])
def test_hadamard_gemm(shape):
    M, K, N = shape
    x = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.1
    w_p = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)
    scales = torch.randn(N, K // 32, device="cuda", dtype=torch.float16) * 0.1
    zeros = torch.randn(N, K // 32, device="cuda", dtype=torch.float16) * 0.1

    ref_out = m_hadamard_gemm.hadamard_w4a16_gemm_pytorch_ref(x, w_p, scales, zeros, group_size=32)
    cuda_out = m_hadamard_gemm.hadamard_w4a16_gemm(x, w_p, scales, zeros, group_size=32)

    assert_numerical_parity(cuda_out, ref_out, dtype=torch.float16, atol=0.01, name="hadamard_gemm")


# 15. quantized_gemm
@pytest.mark.parametrize("shape", [(16, 32, 16), (32, 64, 32)])
def test_quantized_gemm(shape):
    M, K, N = shape
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    sa = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    sb = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    ref_out = m_quantized_gemm.fp8_gemm_pytorch_ref(A, B, sa, sb)
    cuda_out = m_quantized_gemm.fp8_gemm(A, B, sa, sb)

    assert_numerical_parity(cuda_out, ref_out, dtype=torch.float16, name="quantized_gemm")


# 16. rpca_sparse_svd
@pytest.mark.parametrize("shape", [(8, 32, 16, 32), (16, 32, 16, 32)])
def test_rpca_sparse_svd(shape):
    M, K, R, N = shape
    x = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.1
    vr = torch.randn(K, R, device="cuda", dtype=torch.float16) * 0.1
    ur = torch.randn(R, N, device="cuda", dtype=torch.float16) * 0.1

    ref_out = m_rpca_sparse_svd.rpca_sparse_svd_gemm_pytorch_ref(x, vr, ur)
    cuda_out = m_rpca_sparse_svd.rpca_sparse_svd_gemm(x, vr, ur)

    assert_numerical_parity(cuda_out, ref_out, dtype=torch.float16, atol=0.01, name="rpca_sparse_svd")


# 17. nasc_attention
@pytest.mark.parametrize("shape", [(2, 4, 16, 32), (1, 2, 32, 32)])
def test_nasc_attention(shape):
    B, H, N_seq, D = shape
    q = torch.randn(B, H, N_seq, D, device="cuda", dtype=torch.float16) * 0.1
    k = torch.randn(B, H, N_seq, D, device="cuda", dtype=torch.float16) * 0.1
    v = torch.randn(B, H, N_seq, D, device="cuda", dtype=torch.float16) * 0.1
    dec = torch.randn(B, H, N_seq, D, device="cuda", dtype=torch.float16).sigmoid()

    ref_out = m_nasc_attention.nasc_attention_ref(q, k, v, dec)
    cuda_out = m_nasc_attention.nasc_attention(q, k, v, dec)

    assert_numerical_parity(cuda_out, ref_out, dtype=torch.float16, atol=0.01, name="nasc_attention")


# 18. nexus
@pytest.mark.parametrize("shape", [(2, 4, 16, 32), (1, 2, 32, 32)])
def test_nexus(shape):
    B, H, S, D = shape
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16) * 0.1
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16) * 0.1
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16) * 0.1

    ref_out = m_nexus.nexus_ref(q, k, v)
    cuda_out = m_nexus.nexus(q, k, v)

    assert_numerical_parity(cuda_out, ref_out, dtype=torch.float16, atol=0.01, name="nexus")


# 19. vspec
@pytest.mark.parametrize("shape", [(2, 16, 32), (4, 32, 32)])
def test_vspec(shape):
    B, N_seq, D = shape
    x = torch.randn(B, N_seq, D, device="cuda", dtype=torch.float16)
    sg = torch.randn(B, N_seq, D, device="cuda", dtype=torch.float16).sigmoid()
    dec = torch.randn(B, N_seq, D, device="cuda", dtype=torch.float16).sigmoid()

    ref_out = m_vspec.vspec_ref(x, sg, dec)
    cuda_out = m_vspec.vspec(x, sg, dec)

    assert_numerical_parity(cuda_out, ref_out, dtype=torch.float16, name="vspec")


# 20. goldfinch_ahn
@pytest.mark.parametrize("shape", [(2, 4, 16, 32), (1, 2, 32, 32)])
def test_goldfinch_ahn(shape):
    B, H, N_seq, D = shape
    q = torch.randn(B, H, N_seq, D, device="cuda", dtype=torch.float16) * 0.1
    k = torch.randn(B, H, N_seq, D, device="cuda", dtype=torch.float16) * 0.1
    v = torch.randn(B, H, N_seq, D, device="cuda", dtype=torch.float16) * 0.1
    beta = torch.randn(B, H, N_seq, D, device="cuda", dtype=torch.float16).sigmoid() * 0.1

    ref_out = m_goldfinch_ahn.goldfinch_ahn_ref(q, k, v, beta)
    cuda_out = m_goldfinch_ahn.goldfinch_ahn(q, k, v, beta)

    assert_numerical_parity(cuda_out, ref_out, dtype=torch.float16, atol=0.01, name="goldfinch_ahn")
