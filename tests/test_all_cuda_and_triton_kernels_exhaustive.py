# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Exhaustive Multi-Level Parity, Gradient & Shape Verification Suite
for all NeuralOps CUDA C++ and Triton Hot-Path Kernels:
1. selective_scan
2. sliding_window
3. sparse_attention
4. quantized_gemm (w4a16_gemm, fp8_gemm)
"""

import pytest
import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels.selective_scan import (
    selective_scan,
    selective_scan_pytorch_ref,
)
from neuralops.core.cuda_kernels.sliding_window import (
    sliding_window_attn,
    sliding_window_pytorch_ref,
)
from neuralops.core.cuda_kernels.sparse_attention import (
    block_sparse_attn,
    block_sparse_attn_pytorch_ref,
)
from neuralops.core.cuda_kernels.quantized_gemm import (
    w4a16_gemm,
    w4a16_gemm_pytorch_ref,
    fp8_gemm,
    fp8_gemm_pytorch_ref,
)


@pytest.fixture(autouse=True)
def check_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for kernel tests")


@pytest.mark.parametrize("d_state", [8, 16, 32, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_selective_scan_shapes_and_dtypes(d_state, dtype):
    torch.manual_seed(42)
    B, D, L = 2, 8, 32
    u = torch.randn(B, D, L, device="cuda", dtype=dtype)
    delta = torch.randn(B, D, L, device="cuda", dtype=dtype)
    A = torch.randn(D, d_state, device="cuda", dtype=dtype)
    B_t = torch.randn(B, d_state, L, device="cuda", dtype=dtype)
    C = torch.randn(B, d_state, L, device="cuda", dtype=dtype)
    D_param = torch.randn(D, device="cuda", dtype=dtype)

    out_k, state_k = selective_scan(u, delta, A, B_t, C, D_param)
    out_ref, state_ref = selective_scan_pytorch_ref(u, delta, A, B_t, C, D_param)

    tol = 1e-3 if dtype == torch.float32 else (5e-2 if dtype == torch.bfloat16 else 1e-2)
    torch.testing.assert_close(out_k, out_ref, atol=tol, rtol=tol)
    torch.testing.assert_close(state_k, state_ref, atol=tol, rtol=tol)


def test_selective_scan_autograd_gradients():
    torch.manual_seed(42)
    B, D, L, d_state = 2, 4, 16, 16

    u1 = torch.randn(B, D, L, device="cuda", dtype=torch.float32, requires_grad=True)
    delta1 = torch.randn(B, D, L, device="cuda", dtype=torch.float32, requires_grad=True)
    A1 = torch.randn(D, d_state, device="cuda", dtype=torch.float32, requires_grad=True)
    B1 = torch.randn(B, d_state, L, device="cuda", dtype=torch.float32, requires_grad=True)
    C1 = torch.randn(B, d_state, L, device="cuda", dtype=torch.float32, requires_grad=True)
    D1 = torch.randn(D, device="cuda", dtype=torch.float32, requires_grad=True)

    u2 = u1.clone().detach().requires_grad_(True)
    delta2 = delta1.clone().detach().requires_grad_(True)
    A2 = A1.clone().detach().requires_grad_(True)
    B2 = B1.clone().detach().requires_grad_(True)
    C2 = C1.clone().detach().requires_grad_(True)
    D2 = D1.clone().detach().requires_grad_(True)

    out1, _ = selective_scan(u1, delta1, A1, B1, C1, D1)
    loss1 = (out1 ** 2).sum()
    loss1.backward()

    out2, _ = selective_scan_pytorch_ref(u2, delta2, A2, B2, C2, D2)
    loss2 = (out2 ** 2).sum()
    loss2.backward()

    for p1, p2, name in [(u1, u2, "u"), (delta1, delta2, "delta"), (A1, A2, "A"), (B1, B2, "B"), (C1, C2, "C"), (D1, D2, "D")]:
        assert p1.grad is not None, f"{name}.grad is None"
        assert torch.isfinite(p1.grad).all(), f"{name}.grad contains NaN/Inf"
        cos_sim = F.cosine_similarity(p1.grad.flatten().unsqueeze(0), p2.grad.flatten().unsqueeze(0)).item()
        assert cos_sim > 0.995, f"{name}.grad cosine similarity {cos_sim:.4f} < 0.995"


@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("window_size", [16, 64])
@pytest.mark.parametrize("num_sinks", [0, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sliding_window_forward_parity(d, window_size, num_sinks, dtype):
    torch.manual_seed(42)
    B, H, N = 2, 4, 64
    q = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
    k = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
    v = torch.randn(B, H, N, d, device="cuda", dtype=dtype)

    out_k = sliding_window_attn(q, k, v, window_size=window_size, num_sinks=num_sinks)
    out_ref = sliding_window_pytorch_ref(q, k, v, window_size=window_size, num_sinks=num_sinks)

    tol = 1e-3 if dtype == torch.float32 else (5e-2 if dtype == torch.bfloat16 else 1e-2)
    torch.testing.assert_close(out_k, out_ref, atol=tol, rtol=tol)


def test_sliding_window_autograd_gradients():
    torch.manual_seed(42)
    B, H, N, D = 2, 2, 32, 64
    q1 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
    k1 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
    v1 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32, requires_grad=True)

    q2 = q1.clone().detach().requires_grad_(True)
    k2 = k1.clone().detach().requires_grad_(True)
    v2 = v1.clone().detach().requires_grad_(True)

    out1 = sliding_window_attn(q1, k1, v1, window_size=16, num_sinks=4)
    loss1 = (out1 ** 2).sum()
    loss1.backward()

    out2 = sliding_window_pytorch_ref(q2, k2, v2, window_size=16, num_sinks=4)
    loss2 = (out2 ** 2).sum()
    loss2.backward()

    for p1, p2, name in [(q1, q2, "q"), (k1, k2, "k"), (v1, v2, "v")]:
        assert p1.grad is not None
        assert torch.isfinite(p1.grad).all()
        torch.testing.assert_close(p1.grad, p2.grad, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("d", [32, 64, 128])
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_block_sparse_forward_parity(d, block_size, dtype):
    torch.manual_seed(42)
    B, H, N = 2, 2, 64
    q = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
    k = torch.randn(B, H, N, d, device="cuda", dtype=dtype)
    v = torch.randn(B, H, N, d, device="cuda", dtype=dtype)

    num_b = (N + block_size - 1) // block_size
    mask = (torch.rand(num_b, num_b, device="cuda") > 0.5).int()
    mask.fill_diagonal_(1)

    out_k = block_sparse_attn(q, k, v, mask, block_size=block_size)
    out_ref = block_sparse_attn_pytorch_ref(q, k, v, mask, block_size=block_size)

    tol = 1e-3 if dtype == torch.float32 else (5e-2 if dtype == torch.bfloat16 else 1e-2)
    torch.testing.assert_close(out_k, out_ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("shape", [(32, 128, 64), (64, 256, 128), (19, 128, 96)])
@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_w4a16_gemm_parity_and_gradients(shape, group_size, dtype):
    torch.manual_seed(42)
    M, K, N = shape

    A1 = torch.randn(M, K, device="cuda", dtype=dtype, requires_grad=True)
    W_q = torch.randint(0, 255, (K, N // 2), device="cuda", dtype=torch.uint8)
    num_groups = K // group_size
    scales = torch.rand(num_groups, N, device="cuda", dtype=dtype) * 0.1 + 0.01
    zeros = torch.randint(0, 15, (num_groups, N), device="cuda", dtype=torch.int32).to(dtype)

    A2 = A1.clone().detach().requires_grad_(True)

    out1 = w4a16_gemm(A1, W_q, scales, zeros, group_size=group_size)
    loss1 = (out1 ** 2).sum()
    loss1.backward()

    out2 = w4a16_gemm_pytorch_ref(A2, W_q, scales, zeros, group_size=group_size)
    loss2 = (out2 ** 2).sum()
    loss2.backward()

    tol = 1e-3 if dtype == torch.float32 else (5e-2 if dtype == torch.bfloat16 else 1e-2)
    torch.testing.assert_close(out1, out2, atol=tol, rtol=tol)
    assert A1.grad is not None
    assert torch.isfinite(A1.grad).all()
    torch.testing.assert_close(A1.grad, A2.grad, atol=tol * 10, rtol=tol * 10)


@pytest.mark.parametrize("shape", [(32, 64, 48), (64, 128, 64), (19, 96, 48)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fp8_gemm_parity_and_gradients(shape, dtype):
    torch.manual_seed(42)
    M, K, N = shape

    A1 = torch.randn(M, K, device="cuda", dtype=dtype, requires_grad=True)
    B1 = torch.randn(K, N, device="cuda", dtype=dtype, requires_grad=True)
    sa = torch.tensor([0.5], device="cuda", dtype=torch.float32)
    sb = torch.tensor([0.25], device="cuda", dtype=torch.float32)

    A2 = A1.clone().detach().requires_grad_(True)
    B2 = B1.clone().detach().requires_grad_(True)

    out1 = fp8_gemm(A1, B1, sa, sb)
    loss1 = (out1 ** 2).sum()
    loss1.backward()

    out2 = fp8_gemm_pytorch_ref(A2, B2, sa, sb)
    loss2 = (out2 ** 2).sum()
    loss2.backward()

    tol = 1e-3 if dtype == torch.float32 else (5e-2 if dtype == torch.bfloat16 else 1e-2)
    torch.testing.assert_close(out1, out2, atol=tol, rtol=tol)
    assert A1.grad is not None and B1.grad is not None
    assert torch.isfinite(A1.grad).all() and torch.isfinite(B1.grad).all()
    torch.testing.assert_close(A1.grad, A2.grad, atol=tol * 10, rtol=tol * 10)
    torch.testing.assert_close(B1.grad, B2.grad, atol=tol * 10, rtol=tol * 10)
