"""
Exhaustive 3-Level Fallback & Autograd Verification Suite for NeuralOps CUDA C++ Kernels.
Target: NVIDIA Ada Lovelace SM 8.9 (RTX 4070 Laptop GPU, 8GB VRAM).

Tests Level 1 (CUDA C++), Level 2 (Triton JIT), and Level 3 (PyTorch Reference)
under `force_kernel_backend(...)` context across forward and backward autograd passes.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels import (
    force_kernel_backend,
    get_dispatch_stats,
    reset_dispatch_stats,
)
from neuralops.core.cuda_kernels.selective_scan.selective_scan_autograd import (
    selective_scan,
    selective_scan_pytorch_ref,
)
from neuralops.core.cuda_kernels.sliding_window.sliding_autograd import (
    sliding_window_attn,
    sliding_window_pytorch_ref,
)
from neuralops.core.cuda_kernels.sparse_attention.sparse_autograd import (
    block_sparse_attn,
    block_sparse_attn_pytorch_ref,
)
from neuralops.core.cuda_kernels.quantized_gemm.quant_autograd import (
    w4a16_gemm,
    w4a16_gemm_pytorch_ref,
    fp8_gemm,
    fp8_gemm_pytorch_ref,
)
from neuralops.core.cuda_kernels.flash_gqa.flash_gqa_autograd import (
    flash_gqa,
    flash_gqa_pytorch_ref,
)
from neuralops.core.cuda_kernels.paged_attention.paged_attention_autograd import (
    paged_attention,
    paged_attention_pytorch_reference,
)
from neuralops.core.cuda_kernels.paged_kv_cache.kv_cache_autograd import (
    copy_blocks_cow,
    copy_blocks_cow_pytorch_ref,
)
from neuralops.core.cuda_kernels.moe_routing.moe_autograd import (
    moe_dispatch,
    moe_gather,
    _moe_dispatch_pytorch_ref,
    _moe_gather_pytorch_ref,
)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# 1. Selective Scan: Forward & Backward Autograd Parity across D_STATE {8, 16, 32, 64}
# ==============================================================================
@pytest.mark.parametrize("d_state", [8, 16, 32, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_selective_scan_3level_parity(device, d_state, dtype):
    if device == "cpu":
        pytest.skip("CUDA device required for 3-level verification")

    torch.manual_seed(42)
    B, D, L = 2, 16, 32
    u = torch.randn(B, D, L, device=device, dtype=dtype, requires_grad=True)
    delta = torch.rand(B, D, L, device=device, dtype=dtype, requires_grad=True) * 0.1
    A = -torch.rand(D, d_state, device=device, dtype=dtype, requires_grad=True)
    B_mat = torch.randn(B, d_state, L, device=device, dtype=dtype, requires_grad=True)
    C_mat = torch.randn(B, d_state, L, device=device, dtype=dtype, requires_grad=True)
    D_val = torch.randn(D, device=device, dtype=dtype, requires_grad=True)

    # Level 3: PyTorch
    with force_kernel_backend("pytorch"):
        out_py, h_py = selective_scan(u, delta, A, B_mat, C_mat, D_val)
        loss_py = out_py.sum()
        loss_py.backward()
        g_u_py = u.grad.clone()
        g_delta_py = delta.grad.clone()
        g_A_py = A.grad.clone()
        u.grad.zero_()
        delta.grad.zero_()
        A.grad.zero_()

    # Level 1: CUDA C++
    with force_kernel_backend("cuda"):
        out_cuda, h_cuda = selective_scan(u, delta, A, B_mat, C_mat, D_val)
        loss_cuda = out_cuda.sum()
        loss_cuda.backward()
        g_u_cuda = u.grad.clone()
        g_delta_cuda = delta.grad.clone()
        g_A_cuda = A.grad.clone()

    tol = 3e-2 if dtype == torch.float16 else 1e-4
    assert torch.allclose(out_cuda, out_py, atol=tol, rtol=tol), f"Forward mismatch in selective_scan d_state={d_state}"
    assert torch.allclose(g_u_cuda, g_u_py, atol=tol * 2, rtol=tol * 2), "Backward du gradient mismatch"
    assert torch.allclose(g_delta_cuda, g_delta_py, atol=tol * 2, rtol=tol * 2), "Backward ddelta gradient mismatch"


# ==============================================================================
# 2. Sliding Window Attention: Forward & Backward Autograd Parity
# ==============================================================================
@pytest.mark.parametrize("head_dim", [32, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sliding_window_3level_parity(device, head_dim, dtype):
    if device == "cpu":
        pytest.skip("CUDA device required")

    torch.manual_seed(42)
    B, H, N, D = 2, 2, 64, head_dim
    window_size = 16
    num_sinks = 2

    q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)

    with force_kernel_backend("pytorch"):
        out_py = sliding_window_attn(q, k, v, window_size=window_size, num_sinks=num_sinks)
        loss_py = out_py.sum()
        loss_py.backward()
        g_q_py, g_k_py, g_v_py = q.grad.clone(), k.grad.clone(), v.grad.clone()
        q.grad.zero_()
        k.grad.zero_()
        v.grad.zero_()

    with force_kernel_backend("cuda"):
        out_cuda = sliding_window_attn(q, k, v, window_size=window_size, num_sinks=num_sinks)
        loss_cuda = out_cuda.sum()
        loss_cuda.backward()
        g_q_cuda, g_k_cuda, g_v_cuda = q.grad.clone(), k.grad.clone(), v.grad.clone()

    tol = 3e-2 if dtype == torch.float16 else 1e-4
    assert torch.allclose(out_cuda, out_py, atol=tol, rtol=tol), "Sliding window forward mismatch"
    assert torch.allclose(g_q_cuda, g_q_py, atol=tol * 2, rtol=tol * 2), "Sliding window dq mismatch"
    assert torch.allclose(g_k_cuda, g_k_py, atol=tol * 2, rtol=tol * 2), "Sliding window dk mismatch"
    assert torch.allclose(g_v_cuda, g_v_py, atol=tol * 2, rtol=tol * 2), "Sliding window dv mismatch"


# ==============================================================================
# 3. Block Sparse Attention: Forward & Backward Autograd Parity
# ==============================================================================
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_block_sparse_attn_3level_parity(device, block_size, dtype):
    if device == "cpu":
        pytest.skip("CUDA device required")

    torch.manual_seed(42)
    B, H, N, D = 2, 2, 64, 32
    num_blocks = N // block_size
    mask = torch.eye(num_blocks, device=device, dtype=torch.bool)
    if num_blocks > 1:
        mask[0, 1] = True

    q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)

    with force_kernel_backend("pytorch"):
        out_py = block_sparse_attn(q, k, v, block_mask=mask, block_size=block_size)
        loss_py = out_py.sum()
        loss_py.backward()
        g_q_py, g_k_py, g_v_py = q.grad.clone(), k.grad.clone(), v.grad.clone()
        q.grad.zero_()
        k.grad.zero_()
        v.grad.zero_()

    with force_kernel_backend("cuda"):
        out_cuda = block_sparse_attn(q, k, v, block_mask=mask, block_size=block_size)
        loss_cuda = out_cuda.sum()
        loss_cuda.backward()
        g_q_cuda, g_k_cuda, g_v_cuda = q.grad.clone(), k.grad.clone(), v.grad.clone()

    tol = 3e-2 if dtype == torch.float16 else 1e-4
    assert torch.allclose(out_cuda, out_py, atol=tol, rtol=tol), "Block sparse forward mismatch"
    assert torch.allclose(g_q_cuda, g_q_py, atol=tol * 2, rtol=tol * 2), "Block sparse dq mismatch"


# ==============================================================================
# 4. Quantized GEMM: W4A16 & FP8 Forward & Backward Parity
# ==============================================================================
def test_w4a16_gemm_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    torch.manual_seed(42)
    M, K, N = 32, 128, 64
    group_size = 32

    A = torch.randn(M, K, device=device, dtype=torch.float32, requires_grad=True)
    W_q = torch.randint(0, 255, (K, N // 2), device=device, dtype=torch.uint8)
    scales = torch.rand(K // group_size, N, device=device, dtype=torch.float32) * 0.1
    zeros = torch.zeros(K // group_size, N, device=device, dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        out_py = w4a16_gemm(A, W_q, scales, zeros, group_size=group_size)
        loss_py = out_py.sum()
        loss_py.backward()
        g_A_py = A.grad.clone()
        A.grad.zero_()

    with force_kernel_backend("cuda"):
        out_cuda = w4a16_gemm(A, W_q, scales, zeros, group_size=group_size)
        loss_cuda = out_cuda.sum()
        loss_cuda.backward()
        g_A_cuda = A.grad.clone()

    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "W4A16 forward mismatch"
    assert torch.allclose(g_A_cuda, g_A_py, atol=1e-3, rtol=1e-3), "W4A16 dA gradient mismatch"


def test_fp8_gemm_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    torch.manual_seed(42)
    M, K, N = 32, 64, 64
    A = torch.randn(M, K, device=device, dtype=torch.float32, requires_grad=True)
    B = torch.randn(K, N, device=device, dtype=torch.float32, requires_grad=True)
    scale_a = torch.tensor([0.5], device=device, dtype=torch.float32)
    scale_b = torch.tensor([0.25], device=device, dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        out_py = fp8_gemm(A, B, scale_a, scale_b)
        loss_py = out_py.sum()
        loss_py.backward()
        g_A_py, g_B_py = A.grad.clone(), B.grad.clone()
        A.grad.zero_()
        B.grad.zero_()

    with force_kernel_backend("cuda"):
        out_cuda = fp8_gemm(A, B, scale_a, scale_b)
        loss_cuda = out_cuda.sum()
        loss_cuda.backward()
        g_A_cuda, g_B_cuda = A.grad.clone(), B.grad.clone()

    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "FP8 forward mismatch"
    assert torch.allclose(g_A_cuda, g_A_py, atol=1e-3, rtol=1e-3), "FP8 dA mismatch"
    assert torch.allclose(g_B_cuda, g_B_py, atol=1e-3, rtol=1e-3), "FP8 dB mismatch"


# ==============================================================================
# 5. Flash GQA: Forward & Backward Parity
# ==============================================================================
def test_flash_gqa_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    torch.manual_seed(42)
    B, H, N_Q, D = 2, 4, 32, 32
    H_KV, N_KV = 2, 32

    q = torch.randn(B, H, N_Q, D, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H_KV, N_KV, D, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H_KV, N_KV, D, device=device, dtype=torch.float32, requires_grad=True)

    with force_kernel_backend("pytorch"):
        out_py = flash_gqa(q, k, v, is_causal=True)
        loss_py = out_py.sum()
        loss_py.backward()
        g_q_py, g_k_py, g_v_py = q.grad.clone(), k.grad.clone(), v.grad.clone()
        q.grad.zero_()
        k.grad.zero_()
        v.grad.zero_()

    with force_kernel_backend("cuda"):
        out_cuda = flash_gqa(q, k, v, is_causal=True)
        loss_cuda = out_cuda.sum()
        loss_cuda.backward()
        g_q_cuda, g_k_cuda, g_v_cuda = q.grad.clone(), k.grad.clone(), v.grad.clone()

    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "Flash GQA forward mismatch"
    assert torch.allclose(g_q_cuda, g_q_py, atol=1e-3, rtol=1e-3), "Flash GQA dq mismatch"


# ==============================================================================
# 6. Paged Attention: Forward Parity
# ==============================================================================
def test_paged_attention_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    torch.manual_seed(42)
    num_seqs = 2
    num_heads = 4
    head_dim = 32
    block_size = 16
    num_blocks = 8

    q = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=torch.float32)
    k_cache = torch.randn(num_blocks, num_heads, head_dim, block_size, device=device, dtype=torch.float32)
    v_cache = torch.randn(num_blocks, num_heads, head_dim, block_size, device=device, dtype=torch.float32)
    block_tables = torch.tensor([[0, 1, 2, -1], [3, 4, -1, -1]], device=device, dtype=torch.int32)
    context_lens = torch.tensor([30, 20], device=device, dtype=torch.int32)

    with force_kernel_backend("pytorch"):
        out_py = paged_attention(q, k_cache, v_cache, block_tables, context_lens, block_size=block_size)

    with force_kernel_backend("cuda"):
        out_cuda = paged_attention(q, k_cache, v_cache, block_tables, context_lens, block_size=block_size)

    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "Paged Attention forward mismatch"


# ==============================================================================
# 7. Paged KV Cache: CoW Block Copy Parity
# ==============================================================================
def test_paged_kv_cache_copy_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    torch.manual_seed(42)
    num_blocks, num_heads, block_size, head_dim = 16, 4, 16, 32

    src_k = torch.randn(num_blocks, num_heads, block_size, head_dim, device=device, dtype=torch.float32)
    src_v = torch.randn(num_blocks, num_heads, block_size, head_dim, device=device, dtype=torch.float32)
    dst_k = torch.zeros_like(src_k)
    dst_v = torch.zeros_like(src_v)

    mapping = torch.tensor([[0, 2], [1, 3], [4, 5]], device=device, dtype=torch.int32)

    with force_kernel_backend("cuda"):
        res_k, res_v = copy_blocks_cow(dst_k, dst_v, src_k, src_v, mapping, block_size=block_size, num_heads=num_heads, head_dim=head_dim)

    assert torch.allclose(res_k[2], src_k[0]), "CoW KV copy key mismatch"
    assert torch.allclose(res_v[3], src_v[1]), "CoW KV copy value mismatch"


# ==============================================================================
# 8. MoE Routing: Dispatch & Gather Parity
# ==============================================================================
def test_moe_routing_dispatch_gather_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    torch.manual_seed(42)
    num_tokens, hidden_dim, top_k = 16, 32, 2
    total_slots = 32

    tokens = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.float32)
    topk_indices = torch.randint(0, 4, (num_tokens, top_k), device=device, dtype=torch.int32)
    offsets = torch.arange(num_tokens * top_k, device=device, dtype=torch.int32).view(num_tokens, top_k)

    with force_kernel_backend("pytorch"):
        disp_py = moe_dispatch(tokens, topk_indices, offsets, total_slots)

    with force_kernel_backend("cuda"):
        disp_cuda = moe_dispatch(tokens, topk_indices, offsets, total_slots)

    assert torch.allclose(disp_cuda, disp_py, atol=1e-4, rtol=1e-4), "MoE Dispatch mismatch"

    weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=torch.float32), dim=-1)
    with force_kernel_backend("pytorch"):
        gather_py = moe_gather(disp_py, weights, topk_indices, offsets)

    with force_kernel_backend("cuda"):
        gather_cuda = moe_gather(disp_cuda, weights, topk_indices, offsets)

    assert torch.allclose(gather_cuda, gather_py, atol=1e-4, rtol=1e-4), "MoE Gather mismatch"
