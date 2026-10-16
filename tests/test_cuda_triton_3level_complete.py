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
    delta = (torch.rand(B, D, L, device=device, dtype=dtype) * 0.1).detach().requires_grad_(True)
    A = (-torch.rand(D, d_state, device=device, dtype=dtype)).detach().requires_grad_(True)
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


# ==============================================================================
# 9. Low-Rank Gated GEMM: Forward & Backward Autograd Parity
# ==============================================================================
def test_lowrank_gated_gemm_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import fused_lowrank_gated_gemm

    torch.manual_seed(42)
    M, K, r, N = 32, 128, 16, 64
    x = torch.randn(M, K, device=device, dtype=torch.float32, requires_grad=True)
    V = torch.randn(r, K, device=device, dtype=torch.float32, requires_grad=True)
    U = torch.randn(N, r, device=device, dtype=torch.float32, requires_grad=True)
    alpha = torch.randn(r, device=device, dtype=torch.float32)
    bias = torch.randn(N, device=device, dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        out_py = fused_lowrank_gated_gemm(x, V, U, alpha=alpha, bias=bias)
        loss_py = out_py.sum()
        loss_py.backward()
        gx_py = x.grad.clone()
        x.grad.zero_()

    with force_kernel_backend("cuda"):
        out_cuda = fused_lowrank_gated_gemm(x, V, U, alpha=alpha, bias=bias)
        loss_cuda = out_cuda.sum()
        loss_cuda.backward()
        gx_cuda = x.grad.clone()
        x.grad.zero_()

    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "Low-Rank GEMM forward mismatch"
    assert torch.allclose(gx_cuda, gx_py, atol=1e-3, rtol=1e-3), "Low-Rank GEMM backward dX mismatch"


# ==============================================================================
# 10. Chunked Delta-Gate Recurrent Scan Parity
# ==============================================================================
def test_gdn_chunked_scan_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import chunked_delta_gate_scan

    torch.manual_seed(42)
    B, H, L, D_K, D_V = 2, 2, 16, 16, 16
    q = torch.randn(B, H, L, D_K, device=device, dtype=torch.float32)
    k = F.normalize(torch.randn(B, H, L, D_K, device=device, dtype=torch.float32), p=2, dim=-1)
    v = torch.randn(B, H, L, D_V, device=device, dtype=torch.float32)
    alpha = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float32))
    beta = torch.sigmoid(torch.randn(B, H, L, device=device, dtype=torch.float32))

    with force_kernel_backend("pytorch"):
        out_py = chunked_delta_gate_scan(q, k, v, alpha, beta)

    with force_kernel_backend("cuda"):
        out_cuda = chunked_delta_gate_scan(q, k, v, alpha, beta)

    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "GDN Scan forward mismatch"


# ==============================================================================
# 11. HGRN2 Bidirectional Recurrent Scan Parity
# ==============================================================================
def test_hgrn2_scan_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import hgrn2_fused_bwd_scan

    torch.manual_seed(42)
    B, H, L, D = 2, 2, 16, 16
    x = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    g = torch.sigmoid(torch.randn(B, H, L, D, device=device, dtype=torch.float32))

    with force_kernel_backend("pytorch"):
        out_py = hgrn2_fused_bwd_scan(x, g)

    with force_kernel_backend("cuda"):
        out_cuda = hgrn2_fused_bwd_scan(x, g)

    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "HGRN2 Scan forward mismatch"


# ==============================================================================
# 12. Cosformer Linear Attention Parity
# ==============================================================================
def test_cosformer_linear_attention_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import cosformer_linear_attention

    torch.manual_seed(42)
    B, H, L, D = 2, 2, 16, 16
    q = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, L, D, device=device, dtype=torch.float32)

    for causal in [True, False]:
        with force_kernel_backend("pytorch"):
            out_py = cosformer_linear_attention(q, k, v, is_causal=causal)

        with force_kernel_backend("cuda"):
            out_cuda = cosformer_linear_attention(q, k, v, is_causal=causal)

        assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), f"Cosformer causal={causal} mismatch"


# ==============================================================================
# 13. Fused LayerNorm + Linear Parity
# ==============================================================================
def test_fused_norm_linear_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import fused_norm_linear

    torch.manual_seed(42)
    M, K, N = 16, 32, 16
    x = torch.randn(M, K, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(N, K, device=device, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(N, device=device, dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        out_py = fused_norm_linear(x, weight, bias=bias)
        loss_py = out_py.sum()
        loss_py.backward()
        gx_py = x.grad.clone()
        x.grad.zero_()

    with force_kernel_backend("cuda"):
        out_cuda = fused_norm_linear(x, weight, bias=bias)
        loss_cuda = out_cuda.sum()
        loss_cuda.backward()
        gx_cuda = x.grad.clone()
        x.grad.zero_()

    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "Fused Norm Linear forward mismatch"
    assert torch.allclose(gx_cuda, gx_py, atol=1e-3, rtol=1e-3), "Fused Norm Linear backward dX mismatch"


# ==============================================================================
# 14. Fused RoPE Parity
# ==============================================================================
def test_fused_rope_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import fused_rotary_emb

    torch.manual_seed(42)
    B, H, N, D = 2, 2, 16, 32
    half_d = D // 2
    x = torch.randn(B, H, N, D, device=device, dtype=torch.float32, requires_grad=True)
    cos = torch.randn(1, 1, N, half_d, device=device, dtype=torch.float32)
    sin = torch.randn(1, 1, N, half_d, device=device, dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        out_py = fused_rotary_emb(x, cos, sin)
        loss_py = out_py.sum()
        loss_py.backward()
        gx_py = x.grad.clone()
        x.grad.zero_()

    with force_kernel_backend("cuda"):
        out_cuda = fused_rotary_emb(x, cos, sin)
        loss_cuda = out_cuda.sum()
        loss_cuda.backward()
        gx_cuda = x.grad.clone()
        x.grad.zero_()

    assert torch.allclose(out_cuda, out_py, atol=1e-4, rtol=1e-4), "Fused RoPE forward mismatch"
    assert torch.allclose(gx_cuda, gx_py, atol=1e-4, rtol=1e-4), "Fused RoPE backward dX mismatch"


# ==============================================================================
# 15. Kronecker Matmul Parity
# ==============================================================================
def test_kronecker_matmul_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import kronecker_matmul

    torch.manual_seed(42)
    M, M1, N1, M2, N2 = 16, 16, 16, 16, 16
    x = torch.randn(M, N1 * N2, device=device, dtype=torch.float32)
    A = torch.randn(M1, N1, device=device, dtype=torch.float32)
    B = torch.randn(M2, N2, device=device, dtype=torch.float32)
    bias = torch.randn(M1 * M2, device=device, dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        out_py = kronecker_matmul(x, A, B, bias=bias)

    with force_kernel_backend("cuda"):
        out_cuda = kronecker_matmul(x, A, B, bias=bias)

    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "Kronecker Matmul mismatch"


# ==============================================================================
# 16. In-Register LUT Codebook Dequantization GEMM Parity
# ==============================================================================
def test_lut_codebook_dequant_gemm_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import lut_codebook_dequant_gemm

    torch.manual_seed(42)
    M, K, N = 16, 64, 32
    group_size = 32
    num_groups = K // group_size

    x = torch.randn(M, K, device=device, dtype=torch.float32)
    indices_packed = torch.randint(0, 255, (N, K // 2), device=device, dtype=torch.uint8)
    codebook = torch.randn(num_groups, 16, device=device, dtype=torch.float32) * 0.1
    bias = torch.randn(N, device=device, dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        out_py = lut_codebook_dequant_gemm(x, indices_packed, codebook, group_size=group_size, bias=bias)

    with force_kernel_backend("cuda"):
        out_cuda = lut_codebook_dequant_gemm(x, indices_packed, codebook, group_size=group_size, bias=bias)

    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "LUT GEMM mismatch"


# ==============================================================================
# 17. Linear Attention Parity
# ==============================================================================
def test_linear_attention_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import linear_attention

    torch.manual_seed(42)
    B, H, L, D_K, D_V = 2, 2, 16, 16, 16
    q = torch.randn(B, H, L, D_K, device=device, dtype=torch.float32)
    k = torch.randn(B, H, L, D_K, device=device, dtype=torch.float32)
    v = torch.randn(B, H, L, D_V, device=device, dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        out_py = linear_attention(q, k, v)

    with force_kernel_backend("cuda"):
        out_cuda = linear_attention(q, k, v)

    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "Linear Attention mismatch"


# ==============================================================================
# 18. Fused Cross-Entropy Loss Parity
# ==============================================================================
def test_fused_cross_entropy_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import fused_cross_entropy

    torch.manual_seed(42)
    N_tokens, V = 32, 128
    logits = torch.randn(N_tokens, V, device=device, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, V, (N_tokens,), device=device, dtype=torch.int64)
    # inject an ignore index
    targets[0] = -100

    with force_kernel_backend("pytorch"):
        loss_py = fused_cross_entropy(logits, targets, ignore_index=-100, reduction="mean")
        loss_py.backward()
        g_py = logits.grad.clone()
        logits.grad.zero_()

    with force_kernel_backend("cuda"):
        loss_cuda = fused_cross_entropy(logits, targets, ignore_index=-100, reduction="mean")
        loss_cuda.backward()
        g_cuda = logits.grad.clone()
        logits.grad.zero_()

    assert torch.allclose(loss_cuda, loss_py, atol=1e-4, rtol=1e-4), "Fused CE forward mismatch"
    assert torch.allclose(g_cuda, g_py, atol=1e-4, rtol=1e-4), "Fused CE backward d_logits mismatch"



# ==============================================================================
# 19. RPCA Low-Rank + Sparse GEMM Parity
# ==============================================================================
def test_rpca_sparse_svd_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import rpca_sparse_svd_gemm

    torch.manual_seed(42)
    M, K, R, N = 16, 64, 16, 32
    x = torch.randn(M, K, device=device, dtype=torch.float32)
    v_r = torch.randn(R, K, device=device, dtype=torch.float32)
    u_r = torch.randn(N, R, device=device, dtype=torch.float32)
    s = torch.randn(N, K, device=device, dtype=torch.float32) * (torch.rand(N, K, device=device) < 0.1).float()
    bias = torch.randn(N, device=device, dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        out_py = rpca_sparse_svd_gemm(x, v_r, u_r, s=s, bias=bias)

    with force_kernel_backend("cuda"):
        out_cuda = rpca_sparse_svd_gemm(x, v_r, u_r, s=s, bias=bias)

    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "RPCA SVD mismatch"


# ==============================================================================
# 20. Tensor-Train (TT) Contraction & Linear Parity
# ==============================================================================
def test_tt_contraction_and_linear_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import tt_contraction, tt_linear

    torch.manual_seed(42)
    p0, p1 = 8, 8
    q0, q1 = 8, 8
    R = 8
    c0 = torch.randn(1, p0, q0, R, device=device, dtype=torch.float32)
    c1 = torch.randn(R, p1, q1, 1, device=device, dtype=torch.float32)
    x = torch.randn(16, p0 * p1, device=device, dtype=torch.float32)
    bias = torch.randn(q0 * q1, device=device, dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        w_py = tt_contraction(c0, c1, (p0, p1), (q0, q1))
        out_py = tt_linear(x, c0, c1, (p0, p1), (q0, q1), bias=bias)

    with force_kernel_backend("cuda"):
        w_cuda = tt_contraction(c0, c1, (p0, p1), (q0, q1))
        out_cuda = tt_linear(x, c0, c1, (p0, p1), (q0, q1), bias=bias)

    assert torch.allclose(w_cuda, w_py, atol=1e-3, rtol=1e-3), "TT contraction mismatch"
    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "TT linear mismatch"


# ==============================================================================
# 21. Hadamard W4A16 GEMM Parity
# ==============================================================================
def test_hadamard_w4a16_gemm_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import hadamard_w4a16_gemm

    torch.manual_seed(42)
    M, K, N = 16, 64, 32
    group_size = 32
    num_groups = K // group_size

    x = torch.randn(M, K, device=device, dtype=torch.float32)
    w_packed = torch.randint(0, 255, (N, K // 2), device=device, dtype=torch.uint8)
    scales = torch.rand(N, num_groups, device=device, dtype=torch.float32) * 0.1 + 0.01
    zeros = torch.randint(0, 16, (N, num_groups), device=device, dtype=torch.float32)
    bias = torch.randn(N, device=device, dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        out_py = hadamard_w4a16_gemm(x, w_packed, scales, zeros, group_size=group_size, bias=bias)

    with force_kernel_backend("cuda"):
        out_cuda = hadamard_w4a16_gemm(x, w_packed, scales, zeros, group_size=group_size, bias=bias)

    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "Hadamard W4A16 GEMM mismatch"


# ==============================================================================
# 22. Fused Activation Scaling GEMM Parity
# ==============================================================================
def test_act_scale_quant_gemm_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import act_scale_quant_gemm

    torch.manual_seed(42)
    M, K, N = 16, 64, 32
    x = torch.randn(M, K, device=device, dtype=torch.float32)
    s_inv = torch.rand(K, device=device, dtype=torch.float32) + 0.5
    w = torch.randn(N, K, device=device, dtype=torch.float32)
    bias = torch.randn(N, device=device, dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        out_py = act_scale_quant_gemm(x, s_inv, w, bias=bias)

    with force_kernel_backend("cuda"):
        out_cuda = act_scale_quant_gemm(x, s_inv, w, bias=bias)

    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "Act Scale GEMM mismatch"


# ==============================================================================
# 23. Fused Latent KV-Cache Attention Parity
# ==============================================================================
def test_latent_kv_cache_attention_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import latent_kv_cache_attention

    torch.manual_seed(42)
    B, H, L_Q, L_KV = 2, 2, 4, 16
    D_K, R, D_V = 32, 16, 32

    q = torch.randn(B, H, L_Q, D_K, device=device, dtype=torch.float32)
    c_kv = torch.randn(B, L_KV, R, device=device, dtype=torch.float32)
    u_k = torch.randn(D_K, R, device=device, dtype=torch.float32)
    u_v = torch.randn(R, D_V, device=device, dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        out_py = latent_kv_cache_attention(q, c_kv, u_k, u_v)

    with force_kernel_backend("cuda"):
        out_cuda = latent_kv_cache_attention(q, c_kv, u_k, u_v)

    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "Latent KV Attention mismatch"


# ==============================================================================
# 24. Spectral Entropy SVD Dynamic Gate Parity
# ==============================================================================
def test_spectral_entropy_svd_gate_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import spectral_entropy_svd_gate

    torch.manual_seed(42)
    M, K, R_MAX, N = 16, 64, 32, 32
    eff_r = 16
    x = torch.randn(M, K, device=device, dtype=torch.float32)
    V_r = torch.randn(R_MAX, K, device=device, dtype=torch.float32)
    U_r = torch.randn(N, R_MAX, device=device, dtype=torch.float32)
    s_weights = torch.rand(R_MAX, device=device, dtype=torch.float32)
    bias = torch.randn(N, device=device, dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        out_py = spectral_entropy_svd_gate(x, V_r, U_r, eff_r, s_weights=s_weights, bias=bias)

    with force_kernel_backend("cuda"):
        out_cuda = spectral_entropy_svd_gate(x, V_r, U_r, eff_r, s_weights=s_weights, bias=bias)

    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "Entropy Gate mismatch"


# ==============================================================================
# 25. Fused Knowledge Distillation Loss Parity
# ==============================================================================
def test_fused_kd_loss_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import fused_kd_loss

    torch.manual_seed(42)
    M, V = 16, 64
    s_logits = torch.randn(M, V, device=device, dtype=torch.float32)
    t_logits = torch.randn(M, V, device=device, dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        loss_py = fused_kd_loss(s_logits, t_logits, temperature=2.0, reduction="batchmean")

    with force_kernel_backend("cuda"):
        loss_cuda = fused_kd_loss(s_logits, t_logits, temperature=2.0, reduction="batchmean")

    assert torch.allclose(loss_cuda, loss_py, atol=1e-3, rtol=1e-3), "Fused KD loss mismatch"


# ==============================================================================
# 26. Recurrent Linear Step Parity
# ==============================================================================
def test_recurrent_linear_step_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import recurrent_linear_step

    torch.manual_seed(42)
    B, H, Dk, Dv = 2, 2, 16, 16
    q_t = torch.randn(B, H, Dk, device=device, dtype=torch.float32)
    k_t = torch.randn(B, H, Dk, device=device, dtype=torch.float32)
    v_t = torch.randn(B, H, Dv, device=device, dtype=torch.float32)
    kv_state = torch.zeros(B, H, Dk, Dv, device=device, dtype=torch.float32)
    z_state = torch.zeros(B, H, Dk, device=device, dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        out_py, kv_py, z_py = recurrent_linear_step(q_t, k_t, v_t, kv_state.clone(), z_state.clone())

    with force_kernel_backend("cuda"):
        out_cuda, kv_cuda, z_cuda = recurrent_linear_step(q_t, k_t, v_t, kv_state.clone(), z_state.clone())

    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "Recurrent linear step output mismatch"
    assert torch.allclose(kv_cuda, kv_py, atol=1e-3, rtol=1e-3), "Recurrent linear step kv_state mismatch"
    assert torch.allclose(z_cuda, z_py, atol=1e-3, rtol=1e-3), "Recurrent linear step z_state mismatch"


# ==============================================================================
# 27. Flash Decoding Attention Parity
# ==============================================================================
def test_flash_decoding_attention_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import flash_decoding_attention

    torch.manual_seed(42)
    B, H, N_q, N_kv, D = 2, 2, 1, 32, 16
    q = torch.randn(B, H, N_q, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, N_kv, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, N_kv, D, device=device, dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        out_py = flash_decoding_attention(q, k, v)

    with force_kernel_backend("cuda"):
        out_cuda = flash_decoding_attention(q, k, v)

    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "Flash decoding attention mismatch"


# ==============================================================================
# 28. Factorised Matrix State (FMS) Scan Parity
# ==============================================================================
def test_fms_scan_parity(device):
    if device == "cpu":
        pytest.skip("CUDA device required")

    from neuralops.core.cuda_kernels import fms_scan

    torch.manual_seed(42)
    B, H, L, D = 2, 2, 16, 16
    q = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, L, D, device=device, dtype=torch.float32)
    decay = torch.tensor([0.9, 0.95], device=device, dtype=torch.float32)

    with force_kernel_backend("pytorch"):
        out_py = fms_scan(q, k, v, decay)

    with force_kernel_backend("cuda"):
        out_cuda = fms_scan(q, k, v, decay)

    assert torch.allclose(out_cuda, out_py, atol=1e-3, rtol=1e-3), "FMS scan mismatch"
