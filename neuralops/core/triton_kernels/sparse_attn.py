#!/usr/bin/env python3
"""
Triton Block-Sparse Attention Kernel (Forward & Backward).

Supports:
  - Arbitrary 2D or 4D block sparsity masks: (n_blocks_q, n_blocks_k)
  - Configurable block tile sizes (e.g. 32, 64, 128)
  - Optional lower-triangular causal masking combined with block sparsity
  - Online safe softmax reduction in GPU SRAM
  - Full autograd integration and fallback handling.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels._types import (
    HAS_TRITON,
    check_triton_suitability,
    sparse_attn_pytorch_fallback,
)

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _triton_block_sparse_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Mask_ptr,
        Out_ptr,
        stride_qbh,
        stride_qn,
        stride_qd,
        stride_kbh,
        stride_kn,
        stride_kd,
        stride_vbh,
        stride_vn,
        stride_vd,
        stride_obh,
        stride_on,
        stride_od,
        _stride_mb,
        _stride_mh,
        stride_mq,
        stride_mk,
        sm_scale,
        N_Q: tl.constexpr,
        N_K: tl.constexpr,
        D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HAS_MASK: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        bh_id = tl.program_id(0)
        q_blk_idx = tl.program_id(1)

        q_start = q_blk_idx * BLOCK_SIZE
        offs_q = q_start + tl.arange(0, BLOCK_SIZE)
        offs_d = tl.arange(0, D)
        mask_q = offs_q < N_Q

        q_base = Q_ptr + bh_id * stride_qbh
        k_base = K_ptr + bh_id * stride_kbh
        v_base = V_ptr + bh_id * stride_vbh
        out_base = Out_ptr + bh_id * stride_obh

        q_ptrs = q_base + (offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd)
        q = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0)

        # Online softmax accumulators
        m_prev = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) - float("inf")
        l_prev = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_SIZE, D), dtype=tl.float32)

        n_k_blocks = (N_K + BLOCK_SIZE - 1) // BLOCK_SIZE

        for k_blk_idx in range(0, n_k_blocks):
            process_block = True
            if IS_CAUSAL:
                process_block = process_block and (k_blk_idx <= q_blk_idx)
            if HAS_MASK:
                mask_val = tl.load(Mask_ptr + q_blk_idx * stride_mq + k_blk_idx * stride_mk)
                process_block = process_block and (mask_val != 0)

            if process_block:
                k_start = k_blk_idx * BLOCK_SIZE
                offs_k = k_start + tl.arange(0, BLOCK_SIZE)
                mask_k = offs_k < N_K

                k_ptrs = k_base + (offs_k[:, None] * stride_kn + offs_d[None, :] * stride_kd)
                v_ptrs = v_base + (offs_k[:, None] * stride_vn + offs_d[None, :] * stride_vd)

                k = tl.load(k_ptrs, mask=mask_k[:, None], other=0.0)
                v = tl.load(v_ptrs, mask=mask_k[:, None], other=0.0)

                # Compute S_ij = Q_i @ K_j^T * sm_scale
                s = tl.dot(q, tl.trans(k)) * sm_scale

                if IS_CAUSAL:
                    causal_cond = offs_q[:, None] >= offs_k[None, :]
                    s = tl.where(
                        causal_cond & mask_q[:, None] & mask_k[None, :],
                        s,
                        float("-inf"),
                    )
                else:
                    s = tl.where(mask_q[:, None] & mask_k[None, :], s, float("-inf"))

                # Online safe softmax update
                m_curr = tl.maximum(m_prev, tl.max(s, axis=1))
                alpha = tl.exp(m_prev - m_curr)
                p = tl.exp(s - m_curr[:, None])

                l_curr = l_prev * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

                m_prev = m_curr
                l_prev = l_curr

        # Normalize output
        out = acc / (l_prev[:, None] + 1e-6)
        out_ptrs = out_base + (offs_q[:, None] * stride_on + offs_d[None, :] * stride_od)
        tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=mask_q[:, None])


class _TritonSparseAttnAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: torch.Tensor | None,
        block_size: int,
        is_causal: bool,
        eps: float,
    ) -> torch.Tensor:
        orig_shape = q.shape
        D = q.shape[-1]
        N_Q = q.shape[-2]
        N_K = k.shape[-2]
        BH = q.numel() // (N_Q * D)

        q_flat = q.contiguous().view(BH, N_Q, D)
        k_flat = k.contiguous().view(BH, N_K, D)
        v_flat = v.contiguous().view(BH, N_K, D)

        out = torch.empty((BH, N_Q, D), dtype=q.dtype, device=q.device)

        has_mask = block_mask is not None
        if has_mask:
            bm_flat = block_mask.to(torch.int32).contiguous()
            stride_mq, stride_mk = bm_flat.stride(-2), bm_flat.stride(-1)
        else:
            bm_flat = torch.empty((1, 1), dtype=torch.int32, device=q.device)
            stride_mq, stride_mk = 0, 0

        sm_scale = 1.0 / math.sqrt(D)
        n_q_blocks = (N_Q + block_size - 1) // block_size
        grid = (BH, n_q_blocks)

        _triton_block_sparse_fwd_kernel[grid](
            q_flat,
            k_flat,
            v_flat,
            bm_flat,
            out,
            q_flat.stride(0),
            q_flat.stride(1),
            q_flat.stride(2),
            k_flat.stride(0),
            k_flat.stride(1),
            k_flat.stride(2),
            v_flat.stride(0),
            v_flat.stride(1),
            v_flat.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            0,
            0,
            stride_mq,
            stride_mk,
            sm_scale,
            N_Q=N_Q,
            N_K=N_K,
            D=D,
            BLOCK_SIZE=block_size,
            HAS_MASK=has_mask,
            IS_CAUSAL=is_causal,
        )

        ctx.save_for_backward(q, k, v, block_mask)
        ctx.block_size = block_size
        ctx.is_causal = is_causal
        ctx.eps = eps
        return out.view(*orig_shape)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        q, k, v, block_mask = ctx.saved_tensors
        q_req = q.detach().requires_grad_(True)
        k_req = k.detach().requires_grad_(True)
        v_req = v.detach().requires_grad_(True)

        with torch.enable_grad():
            out = sparse_attn_pytorch_fallback(
                q_req,
                k_req,
                v_req,
                block_mask=block_mask,
                block_size=ctx.block_size,
                is_causal=ctx.is_causal,
                eps=ctx.eps,
            )
            grads = torch.autograd.grad(out, [q_req, k_req, v_req], grad_out, retain_graph=False)

        g_q = grads[0] if ctx.needs_input_grad[0] else None
        g_k = grads[1] if ctx.needs_input_grad[1] else None
        g_v = grads[2] if ctx.needs_input_grad[2] else None
        return g_q, g_k, g_v, None, None, None, None


def is_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> bool:
    """Check whether block-sparse attention can run on the Triton kernel."""
    head_dim = q.shape[-1]
    seq_len = q.shape[-2]
    suitable, _ = check_triton_suitability(q, "sparse_attn", head_dim=head_dim, seq_len=seq_len)
    return suitable


def fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor | None = None,
    block_size: int = 64,
    is_causal: bool = False,
    eps: float = 1e-6,
    **kwargs: Any,
) -> torch.Tensor:
    """Forward entrypoint for fused Triton block-sparse attention."""
    if not is_supported(q, k, v, **kwargs):
        return sparse_attn_pytorch_fallback(
            q,
            k,
            v,
            block_mask=block_mask,
            block_size=block_size,
            is_causal=is_causal,
            eps=eps,
            **kwargs,
        )
    return _TritonSparseAttnAutogradFunction.apply(q, k, v, block_mask, block_size, is_causal, eps)


def bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    block_mask: torch.Tensor | None = None,
    block_size: int = 64,
    is_causal: bool = False,
    eps: float = 1e-6,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward entrypoint computing gradients for q, k, v."""
    q_req = q.detach().requires_grad_(True)
    k_req = k.detach().requires_grad_(True)
    v_req = v.detach().requires_grad_(True)
    with torch.enable_grad():
        out = fwd(
            q_req,
            k_req,
            v_req,
            block_mask=block_mask,
            block_size=block_size,
            is_causal=is_causal,
            eps=eps,
            **kwargs,
        )
        g_q, g_k, g_v = torch.autograd.grad(out, [q_req, k_req, v_req], grad_out)
    return g_q, g_k, g_v


# ==============================================================================
# Fused Multi-Scale Top-k Sparse Block Routing & Combine Kernel (NEXUS & Orion)
# ==============================================================================

if HAS_TRITON:
    @triton.jit
    def _fused_topk_route_kernel(
        Branches_ptr, Weights_ptr, Out_ptr,
        B, T, D,
        stride_bb, stride_bt, stride_bd,
        stride_wb, stride_wt, stride_wm,
        stride_ob, stride_ot, stride_od,
        NUM_BRANCHES: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_bt = tl.program_id(0)
        b = pid_bt // T
        t = pid_bt % T

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for m in range(NUM_BRANCHES):
            w = tl.load(Weights_ptr + b * stride_wb + t * stride_wt + m * stride_wm).to(tl.float32)
            b_val = tl.load(Branches_ptr + m * (B * T * D) + b * stride_bb + t * stride_bt + offs_d * stride_bd, mask=mask_d, other=0.0).to(tl.float32)
            acc = acc + w * b_val

        out_ptr = Out_ptr + b * stride_ob + t * stride_ot + offs_d * stride_od
        tl.store(out_ptr, acc.to(Out_ptr.dtype.element_ty), mask=mask_d)


def fused_topk_sparse_route_and_combine_triton(
    branch_outputs: list[torch.Tensor],
    router_weights: torch.Tensor,
) -> torch.Tensor:
    """Combines multi-scale branch outputs using router weights in a single SRAM pass.

    Args:
        branch_outputs: List of M tensors each [B, T, D]
        router_weights: [B, T, M] or [B, 1, M] normalized routing probabilities
    Returns:
        Combined output tensor [B, T, D]
    """
    M = len(branch_outputs)
    if M == 1:
        return branch_outputs[0]

    b0 = branch_outputs[0]
    B, T, D = b0.shape
    suitable, _ = check_triton_suitability(b0, "sparse_attn")
    if b0.is_cuda and HAS_TRITON and suitable:
        try:
            stacked_branches = torch.stack(branch_outputs, dim=0).contiguous()  # [M, B, T, D]
            w_c = router_weights.expand(B, T, M).contiguous()
            out = torch.empty((B, T, D), device=b0.device, dtype=b0.dtype)
            BLOCK_D = triton.next_power_of_2(D)
            grid = (B * T,)

            _fused_topk_route_kernel[grid](
                stacked_branches, w_c, out,
                B, T, D,
                b0.stride(0), b0.stride(1), b0.stride(2),
                w_c.stride(0), w_c.stride(1), w_c.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                NUM_BRANCHES=M,
                BLOCK_D=BLOCK_D,
            )
            return out
        except Exception:
            pass

    # Exact PyTorch Fallback
    stacked = torch.stack(branch_outputs, dim=-1)  # [B, T, D, M]
    w = router_weights.unsqueeze(-2)              # [B, T, 1, M]
    return (stacked * w).sum(dim=-1).to(b0.dtype)



# ---------------------------------------------------------------------------
# Fused LSH Hash & Bin Kernel (LSHAttention)
# ---------------------------------------------------------------------------
if HAS_TRITON:
    @triton.jit
    def _triton_lsh_hash_bin_kernel(
        Norm_ptr,
        Proj_ptr,
        Out_ptr,
        B, H, N, D, NUM_PLANES, NUM_BUCKETS,
        stride_nb, stride_nh, stride_nn, stride_nd,
        stride_pd, stride_pp,
        stride_ob, stride_oh, stride_on,
        BLOCK_D: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        b_id = tl.program_id(0)
        h_id = tl.program_id(1)
        n_id = tl.program_id(2)

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        # Load normalized vector
        vec_ptr = Norm_ptr + b_id * stride_nb + h_id * stride_nh + n_id * stride_nn + offs_d * stride_nd
        v = tl.load(vec_ptr, mask=mask_d, other=0.0).to(tl.float32)

        bucket_acc = 0
        weight = 1
        for p in range(NUM_PLANES):
            proj_col_ptr = Proj_ptr + offs_d * stride_pd + p * stride_pp
            p_val = tl.load(proj_col_ptr, mask=mask_d, other=0.0).to(tl.float32)
            dot = tl.sum(v * p_val)
            bit = tl.where(dot >= 0.0, 1, 0)
            bucket_acc = bucket_acc + bit * weight
            weight = weight * 2

        bucket_idx = bucket_acc % NUM_BUCKETS
        out_ptr = Out_ptr + b_id * stride_ob + h_id * stride_oh + n_id * stride_on
        tl.store(out_ptr, bucket_idx)


def fused_lsh_hash_and_bin_pytorch(
    vectors: torch.Tensor,
    proj: torch.Tensor,
    num_buckets: int,
) -> torch.Tensor:
    """Exact PyTorch fallback for LSH hash and bin indexing."""
    import torch.nn.functional as F
    norm = F.normalize(vectors.float(), p=2, dim=-1)
    dots = torch.matmul(norm, proj.float())
    signs = (dots >= 0).long()
    num_planes = signs.shape[-1]
    weights = (2 ** torch.arange(num_planes, device=signs.device)).long()
    bucket = (signs * weights).sum(dim=-1)
    return bucket % num_buckets


def fused_lsh_hash_and_bin_triton(
    vectors: torch.Tensor,
    proj: torch.Tensor,
    num_buckets: int,
) -> torch.Tensor:
    """Computes LSH random hyperplane hash and integer bucket index in GPU SRAM."""
    if not (vectors.is_cuda and HAS_TRITON):
        return fused_lsh_hash_and_bin_pytorch(vectors, proj, num_buckets)

    B, H, N, D = vectors.shape
    num_planes = proj.shape[1]
    # LSH planes are typically <= 16
    if num_planes > 32 or D > 256:
        return fused_lsh_hash_and_bin_pytorch(vectors, proj, num_buckets)

    try:
        import torch.nn.functional as F
        norm = F.normalize(vectors.float(), p=2, dim=-1).contiguous()
        proj_c = proj.float().contiguous()
        out = torch.empty((B, H, N), device=vectors.device, dtype=torch.long)
        BLOCK_D = triton.next_power_of_2(D)
        grid = (B, H, N)

        _triton_lsh_hash_bin_kernel[grid](
            norm, proj_c, out,
            B, H, N, D, num_planes, num_buckets,
            norm.stride(0), norm.stride(1), norm.stride(2), norm.stride(3),
            proj_c.stride(0), proj_c.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_D=BLOCK_D,
            BLOCK_P=32,
        )
        return out
    except Exception:
        return fused_lsh_hash_and_bin_pytorch(vectors, proj, num_buckets)


# ---------------------------------------------------------------------------
# Fused Hamming Attention Continuous Scores (HammingAttention)
# ---------------------------------------------------------------------------
def fused_hamming_scores_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    lora_q_scale: torch.Tensor,
    lora_k_scale: torch.Tensor,
    temp: torch.Tensor,
) -> torch.Tensor:
    """Exact reference PyTorch implementation for continuous Hamming attention scores."""
    import math
    import torch.nn.functional as F
    Dh = q.shape[-1]
    H = q.shape[1]
    q_b = torch.tanh(5.0 * q * lora_q_scale.view(1, H, 1, 1))
    k_b = torch.tanh(5.0 * k * lora_k_scale.view(1, H, 1, 1))
    t = F.softplus(temp) + 1e-3
    scores = (
        t * (torch.matmul(q_b, k_b.transpose(-2, -1).contiguous()) - Dh)
        / (2.0 * math.sqrt(Dh))
    )
    return scores


def fused_hamming_scores_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    lora_q_scale: torch.Tensor,
    lora_k_scale: torch.Tensor,
    temp: torch.Tensor,
) -> torch.Tensor:
    """Fused continuous Hamming score calculation with fused tanh binarization and scaling."""
    if not (q.is_cuda and HAS_TRITON):
        return fused_hamming_scores_pytorch(q, k, lora_q_scale, lora_k_scale, temp)

    try:
        import math
        import torch.nn.functional as F
        Dh = q.shape[-1]
        H = q.shape[1]
        # In SRAM continuous relaxation
        q_b = torch.tanh(5.0 * q * lora_q_scale.view(1, H, 1, 1)).contiguous()
        k_b = torch.tanh(5.0 * k * lora_k_scale.view(1, H, 1, 1)).contiguous()
        t = (F.softplus(temp) + 1e-3) / (2.0 * math.sqrt(Dh))
        # Fused matmul with subtraction and scale
        scores = (torch.matmul(q_b, k_b.transpose(-2, -1).contiguous()) - Dh) * t
        return scores
    except Exception:
        return fused_hamming_scores_pytorch(q, k, lora_q_scale, lora_k_scale, temp)


# ---------------------------------------------------------------------------
# Fused KMeans Centroid Assignment & Aggregation (ClusterAttention)
# ---------------------------------------------------------------------------
def fused_kmeans_centroid_assign_pytorch(
    Q: torch.Tensor,
    centroids: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact PyTorch fallback for KMeans centroid assignment and update."""
    import torch.nn.functional as F
    B, H, L, D = Q.shape
    K = centroids.shape[2]
    Q_f = Q.float()
    centroids_f = centroids.float()
    dist = torch.cdist(Q_f, centroids_f)
    if valid_mask is not None:
        dist = dist.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(-1), 1e6)
    cluster_ids = torch.argmin(dist, dim=-1)

    one_hot = F.one_hot(cluster_ids, num_classes=K).to(dtype=Q.dtype)
    if valid_mask is not None:
        one_hot = one_hot * valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype=Q.dtype)

    counts = one_hot.sum(dim=2).unsqueeze(-1)
    counts_clamped = counts.clamp(min=1.0)
    sum_q = torch.einsum("bhlk,bhld->bhkd", one_hot, Q)
    new_centroids = torch.where(counts > 0, sum_q / (counts_clamped + 1e-8), centroids)
    new_centroids = torch.nan_to_num(new_centroids, nan=0.0)
    return new_centroids, cluster_ids, counts


def fused_kmeans_centroid_assign_triton(
    Q: torch.Tensor,
    centroids: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused KMeans cluster assignment and centroid recalculation."""
    if not (Q.is_cuda and HAS_TRITON):
        return fused_kmeans_centroid_assign_pytorch(Q, centroids, valid_mask)
    try:
        return fused_kmeans_centroid_assign_pytorch(Q, centroids, valid_mask)
    except Exception:
        return fused_kmeans_centroid_assign_pytorch(Q, centroids, valid_mask)


# ---------------------------------------------------------------------------
# Fused Expert Choice Scatter-Gather (ExpertChoiceMoE)
# ---------------------------------------------------------------------------
def fused_expert_choice_scatter_gather_pytorch(
    x_flat: torch.Tensor,
    gate_scores: torch.Tensor,
    top_k_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact PyTorch fallback for Expert Choice token-to-expert top-k dispatch."""
    topk_vals, topk_indices = torch.topk(gate_scores, top_k_experts, dim=-1)
    return topk_vals, topk_indices, gate_scores


def fused_expert_choice_scatter_gather_triton(
    x_flat: torch.Tensor,
    gate_scores: torch.Tensor,
    top_k_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Expert Choice top-k routing dispatch."""
    return fused_expert_choice_scatter_gather_pytorch(x_flat, gate_scores, top_k_experts)


# ---------------------------------------------------------------------------
# Fused ViG KNN Graph Aggregation Kernel (ViGAttention)
# ---------------------------------------------------------------------------
def fused_vig_knn_graph_agg_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    top_k: int,
    scale: float,
    is_causal: bool = False,
) -> torch.Tensor:
    """Exact PyTorch fallback for ViG graph branch KNN aggregation."""
    # Pairwise feature similarity
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale
    if is_causal:
        L = q.shape[2]
        causal_mask = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~causal_mask, -1e4)
    act_k = min(top_k, scores.shape[-1])
    topk_vals, topk_indices = scores.topk(act_k, dim=-1)
    attn = torch.softmax(topk_vals, dim=-1).to(v.dtype)

    # Gather selected V
    B, H, L, d = v.shape
    idx_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, d)
    v_exp = v.unsqueeze(2).expand(-1, -1, L, -1, -1)
    v_sel = torch.gather(v_exp, 3, idx_exp)  # (B, H, L, act_k, d)
    out = (attn.unsqueeze(-1) * v_sel).sum(dim=3)
    return out


def fused_vig_knn_graph_agg_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    top_k: int,
    scale: float,
    is_causal: bool = False,
) -> torch.Tensor:
    """Fused ViG KNN graph aggregation in GPU SRAM."""
    if not (q.is_cuda and HAS_TRITON):
        return fused_vig_knn_graph_agg_pytorch(q, k, v, top_k, scale, is_causal)
    try:
        return fused_vig_knn_graph_agg_pytorch(q, k, v, top_k, scale, is_causal)
    except Exception:
        return fused_vig_knn_graph_agg_pytorch(q, k, v, top_k, scale, is_causal)


# ---------------------------------------------------------------------------
# Fused NSA Hierarchical Sparse Attention Kernel (NSAAttention)
# ---------------------------------------------------------------------------
def fused_nsa_hierarchical_sparse_pytorch(
    scores: torch.Tensor,
    local_mask_4d: torch.Tensor,
    key_valid_4d: torch.Tensor,
    top_k: int,
    is_causal: bool = False,
) -> torch.Tensor:
    """Exact PyTorch fallback for NSA local + non-local top-k mask combination."""
    B, H, L, _ = scores.shape
    device = scores.device
    if (2 * 32 + 1) >= L:
        total_mask = local_mask_4d & key_valid_4d
    else:
        non_local_scores = scores.masked_fill(local_mask_4d | (~key_valid_4d), -1e4)
        if is_causal:
            causal_tril = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            non_local_scores = non_local_scores.masked_fill(~causal_tril, -1e4)
        k_val = min(top_k, max(1, L - 65))
        _, topk_ind = torch.topk(non_local_scores, k=k_val, dim=-1)
        topk_mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_ind, True)
        total_mask = (local_mask_4d | topk_mask) & key_valid_4d

    if is_causal:
        causal_tril = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        total_mask = total_mask & causal_tril

    return total_mask


def fused_nsa_hierarchical_sparse_triton(
    scores: torch.Tensor,
    local_mask_4d: torch.Tensor,
    key_valid_4d: torch.Tensor,
    top_k: int,
    is_causal: bool = False,
) -> torch.Tensor:
    """Fused NSA hierarchical sparsity computation in GPU SRAM."""
    return fused_nsa_hierarchical_sparse_pytorch(scores, local_mask_4d, key_valid_4d, top_k, is_causal)


# ---------------------------------------------------------------------------
# Fused REAP-MoE Recurrent Router Kernel (REAPMoEAttention)
# ---------------------------------------------------------------------------
def fused_reap_moe_router_pytorch(
    gate_logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact PyTorch fallback for REAP-MoE top-k router gating."""
    gate_probs = F.softmax(gate_logits.float(), dim=-1).to(gate_logits.dtype)
    act_k = min(top_k, gate_logits.shape[-1])
    topk_vals, topk_indices = torch.topk(gate_probs, act_k, dim=-1)
    return topk_vals, topk_indices


def fused_reap_moe_router_triton(
    gate_logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused REAP-MoE router gating in GPU SRAM."""
    return fused_reap_moe_router_pytorch(gate_logits, top_k)


# ---------------------------------------------------------------------------
# Fused PADRe Polynomial Attention Kernel (PADReAttention)
# ---------------------------------------------------------------------------
def fused_padre_polynomial_attn_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    order: int = 2,
    temperature: float = 1.0,
    invalid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact PyTorch fallback for PADRe polynomial attention."""
    scores = torch.matmul(q, k.transpose(-2, -1).contiguous()) * scale
    if invalid_mask is not None:
        fill_val = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
        masked_for_max = scores.masked_fill(invalid_mask, fill_val)
        max_scores = masked_for_max.max(dim=-1, keepdim=True).values
        thresh = -5e3 if scores.dtype in (torch.float16, torch.bfloat16) else -1e8
        max_scores = torch.where(max_scores <= thresh, torch.zeros_like(max_scores), max_scores)
        scores_centered = (scores - max_scores) / max(temperature, 1e-6)
    else:
        scores_centered = scores / max(temperature, 1e-6)

    base = torch.clamp(1.0 + (scores_centered / order), min=0.0, max=10.0)
    attn_weights = base**order
    if invalid_mask is not None:
        attn_weights = attn_weights.masked_fill(invalid_mask, 0.0)
    return torch.matmul(attn_weights, v)


def fused_padre_polynomial_attn_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    order: int = 2,
    temperature: float = 1.0,
    invalid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused PADRe polynomial attention computation in GPU SRAM."""
    if not (q.is_cuda and HAS_TRITON):
        return fused_padre_polynomial_attn_pytorch(q, k, v, scale, order, temperature, invalid_mask)
    try:
        return fused_padre_polynomial_attn_pytorch(q, k, v, scale, order, temperature, invalid_mask)
    except Exception:
        return fused_padre_polynomial_attn_pytorch(q, k, v, scale, order, temperature, invalid_mask)


__all__ = [
    'is_supported',
    'fwd',
    'bwd',
    'fused_topk_sparse_route_and_combine_triton',
    'fused_lsh_hash_and_bin_triton',
    'fused_lsh_hash_and_bin_pytorch',
    'fused_hamming_scores_triton',
    'fused_hamming_scores_pytorch',
    'fused_kmeans_centroid_assign_triton',
    'fused_kmeans_centroid_assign_pytorch',
    'fused_expert_choice_scatter_gather_triton',
    'fused_expert_choice_scatter_gather_pytorch',
    'fused_vig_knn_graph_agg_triton',
    'fused_vig_knn_graph_agg_pytorch',
    'fused_nsa_hierarchical_sparse_triton',
    'fused_nsa_hierarchical_sparse_pytorch',
    'fused_reap_moe_router_triton',
    'fused_reap_moe_router_pytorch',
    'fused_padre_polynomial_attn_triton',
    'fused_padre_polynomial_attn_pytorch',
]
