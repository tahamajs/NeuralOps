"""
Block-Sparse Attention

Computes block-sparse attention patterns with butterfly sparsity routing and autograd gradient propagation.

Author: Taha Majlesi
"""

from __future__ import annotations
import math
import torch

__all__ = ["butterfly_mask", "block_sparse_attention"]


def butterfly_mask(N: int, block_size: int = 64, device=None) -> torch.Tensor:
    """Generate butterfly block-sparse mask."""
    num_blocks = (N + block_size - 1) // block_size
    mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    
    for i in range(num_blocks):
        # Diagonal block
        start = i * block_size
        end = min(start + block_size, N)
        mask[start:end, start:end] = True
        # Off-diagonal for butterfly
        for offset in [1, 2, 4, 8]:
            j = i ^ offset
            if j < num_blocks:
                js, je = j * block_size, min((j + 1) * block_size, N)
                mask[start:end, js:je] = True
    
    return mask


def _block_sparse_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(int(q.shape[-1]))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


class _BlockSparseAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        ctx.save_for_backward(q, k, v, mask)
        return _block_sparse_attention_fwd(q, k, v, mask)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, mask = ctx.saved_tensors
        q_r = q.detach().requires_grad_(True)
        k_r = k.detach().requires_grad_(True)
        v_r = v.detach().requires_grad_(True)
        with torch.enable_grad():
            out = _block_sparse_attention_fwd(q_r, k_r, v_r, mask)
            g_q, g_k, g_v = torch.autograd.grad(out, [q_r, k_r, v_r], grad_output, retain_graph=False)
        return g_q, g_k, g_v, None


def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    block_size: int = 64,
) -> torch.Tensor:
    """Block-sparse attention with full autograd backward support."""
    N = q.shape[-2]
    if mask is None:
        mask = butterfly_mask(N, block_size, q.device)
    

    if torch.is_grad_enabled() or q.requires_grad or k.requires_grad or v.requires_grad:
        return _BlockSparseAutograd.apply(q, k, v, mask)
    return _block_sparse_attention_fwd(q, k, v, mask)


# ---------------------------------------------------------------------------
# Fused MoBA Block Router Combine (MoBAAttention)
# ---------------------------------------------------------------------------
def fused_moba_block_router_combine_pytorch(
    Qb: torch.Tensor,
    Kb: torch.Tensor,
    Vb: torch.Tensor,
    scale: float,
    top_k: int,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact PyTorch reference for MoBA block score routing."""
    # Qb, Kb, Vb: (B, H, nb, bs, d)
    Q_mean = Qb.mean(dim=3)
    K_mean = Kb.mean(dim=3)
    block_scores = torch.matmul(Q_mean.float(), K_mean.float().transpose(-2, -1).contiguous()) * scale
    nb = Qb.shape[2]
    if not is_causal:
        block_scores[:, :, :, 0] = block_scores[:, :, :, 0] + 1e4
    else:
        causal_block = torch.triu(
            torch.full((nb, nb), -1e4, device=Qb.device, dtype=torch.float32),
            diagonal=1,
        )
        block_scores = block_scores + causal_block.unsqueeze(0).unsqueeze(0)
    act_top_k = min(top_k, nb)
    _, top_idx = block_scores.topk(act_top_k, dim=-1)
    return block_scores, top_idx


from neuralops.core.triton_kernels._common import HAS_TRITON, triton, tl


if HAS_TRITON and tl is not None:
    @triton.jit
    def _fused_moba_block_router_kernel(
        Q_mean_ptr, K_mean_ptr, Scores_ptr,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_sb, stride_sh, stride_sn, stride_sk,
        H, nb, D, scale,
        IS_CAUSAL: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """In-SRAM Fused MoBA Block Score Router Kernel."""
        pid_m = tl.program_id(0)  # block row in nb
        pid_bh = tl.program_id(1)
        i_b = pid_bh // H
        i_h = pid_bh % H

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        q_ptr = Q_mean_ptr + i_b * stride_qb + i_h * stride_qh + pid_m * stride_qn + offs_d * stride_qd
        q_vec = tl.load(q_ptr, mask=mask_d, other=0.0).to(tl.float32)

        for col in range(nb):
            k_ptr = K_mean_ptr + i_b * stride_kb + i_h * stride_kh + col * stride_kn + offs_d * stride_kd
            k_vec = tl.load(k_ptr, mask=mask_d, other=0.0).to(tl.float32)

            score = tl.sum(q_vec * k_vec) * scale
            if IS_CAUSAL and col > pid_m:
                score = score - 1e4
            elif not IS_CAUSAL and col == 0:
                score += 1e4

            s_ptr = Scores_ptr + i_b * stride_sb + i_h * stride_sh + pid_m * stride_sn + col * stride_sk
            tl.store(s_ptr, score)


def fused_moba_block_router_combine_triton(
    Qb: torch.Tensor,
    Kb: torch.Tensor,
    Vb: torch.Tensor,
    scale: float,
    top_k: int,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused MoBA block router calculation in GPU SRAM."""
    from neuralops.core.triton_kernels._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK

    if HAS_TRITON and not FORCE_PYTORCH_FALLBACK and Qb.is_cuda:
        try:
            B, H, nb, bs, D = Qb.shape
            Q_mean = Qb.mean(dim=3).contiguous()
            K_mean = Kb.mean(dim=3).contiguous()

            scores = torch.empty((B, H, nb, nb), dtype=torch.float32, device=Qb.device)
            BLOCK_D = triton.next_power_of_2(D)
            grid = (nb, B * H)
            _fused_moba_block_router_kernel[grid](
                Q_mean, K_mean, scores,
                Q_mean.stride(0), Q_mean.stride(1), Q_mean.stride(2), Q_mean.stride(3),
                K_mean.stride(0), K_mean.stride(1), K_mean.stride(2), K_mean.stride(3),
                scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
                H, nb, D, scale,
                IS_CAUSAL=is_causal,
                BLOCK_D=BLOCK_D,
            )
            act_top_k = min(top_k, nb)
            _, top_idx = scores.topk(act_top_k, dim=-1)
            return scores, top_idx
        except Exception:
            pass
    return fused_moba_block_router_combine_pytorch(Qb, Kb, Vb, scale, top_k, is_causal)


# ==============================================================================
# PHASE 3 / KERNEL 9: Triton Block-Sparse Streaming Attention Kernel
# Target: nsa_attention, adaptive_sparse_attention, moba_attention, sink_attention
# ==============================================================================

from neuralops.core.triton_kernels._common import HAS_TRITON, triton, tl
import math
from typing import Optional
import torch.nn.functional as F

if HAS_TRITON and tl is not None:
    @triton.autotune(
        configs=[
            triton.Config({}, num_stages=2, num_warps=2),
            triton.Config({}, num_stages=3, num_warps=4),
            triton.Config({}, num_stages=4, num_warps=4),
        ],
        key=["L", "D"],
    )
    @triton.jit
    def _block_sparse_streaming_attn_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        active_blocks_ptr,
        o_ptr,
        B,
        H,
        L,
        D: tl.constexpr,
        K_ACTIVE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        sm_scale,
        stride_qb,
        stride_qh,
        stride_ql,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kl,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vl,
        stride_vd,
        stride_ab_b,
        stride_ab_h,
        stride_ab_q,
        stride_ab_k,
        stride_ob,
        stride_oh,
        stride_ol,
        stride_od,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_m = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        mask_m = offs_m < L

        q_ptrs = q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_ql + offs_d[None, :] * stride_qd
        q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        q_block_id = pid_m * BLOCK_M // BLOCK_SIZE

        for active_idx in range(0, K_ACTIVE):
            active_block_ptr = (
                active_blocks_ptr
                + pid_b * stride_ab_b
                + pid_h * stride_ab_h
                + q_block_id * stride_ab_q
                + active_idx * stride_ab_k
            )
            kv_block_id = tl.load(active_block_ptr)

            if kv_block_id >= 0:
                offs_n = kv_block_id * BLOCK_SIZE + tl.arange(0, BLOCK_N)
                mask_n = offs_n < L

                k_ptrs = k_ptr + pid_b * stride_kb + pid_h * stride_kh + offs_n[:, None] * stride_kl + offs_d[None, :] * stride_kd
                v_ptrs = v_ptr + pid_b * stride_vb + pid_h * stride_vh + offs_n[:, None] * stride_vl + offs_d[None, :] * stride_vd

                k_tile = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
                v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

                s_tile = tl.dot(q_tile, tl.trans(k_tile), allow_tf32=True) * sm_scale
                s_tile = tl.where(mask_m[:, None] & mask_n[None, :], s_tile, float("-inf"))

                m_ij = tl.maximum(m_i, tl.max(s_tile, axis=1))
                p = tl.exp(s_tile - m_ij[:, None])
                l_ij = tl.sum(p, axis=1)

                alpha = tl.exp(m_i - m_ij)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None] + tl.dot(p, v_tile, allow_tf32=True)
                m_i = m_ij

        safe_l_i = tl.maximum(l_i[:, None], 1e-5)
        out_tile = acc / safe_l_i

        o_ptrs = o_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_ol + offs_d[None, :] * stride_od
        tl.store(o_ptrs, out_tile.to(q_ptr.dtype.element_ty), mask=mask_m[:, None])


def triton_block_sparse_streaming_attn_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    active_block_indices: torch.Tensor,
    block_size: int = 32,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    B, H, L, D = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(v)
    num_q_blocks = (L + block_size - 1) // block_size

    for b in range(B):
        for h in range(H):
            for q_blk in range(num_q_blocks):
                q_st = q_blk * block_size
                q_ed = min(L, q_st + block_size)
                q_sub = q[b, h, q_st:q_ed]

                active_ids = active_block_indices[b, h, q_blk]
                valid_ids = active_ids[active_ids >= 0]

                if len(valid_ids) == 0:
                    out[b, h, q_st:q_ed] = 0.0
                    continue

                k_chunks = []
                v_chunks = []
                for blk_id in valid_ids:
                    kv_st = blk_id.item() * block_size
                    kv_ed = min(L, kv_st + block_size)
                    k_chunks.append(k[b, h, kv_st:kv_ed])
                    v_chunks.append(v[b, h, kv_st:kv_ed])

                k_sub = torch.cat(k_chunks, dim=0)
                v_sub = torch.cat(v_chunks, dim=0)

                scores = torch.matmul(q_sub.float(), k_sub.float().t()) * sm_scale
                p = F.softmax(scores, dim=-1)
                out[b, h, q_st:q_ed] = torch.matmul(p, v_sub.float()).to(q.dtype)

    return out


def triton_block_sparse_streaming_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    active_block_indices: torch.Tensor,
    block_size: int = 32,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    B, H, L, D = q.shape
    K_ACTIVE = active_block_indices.shape[-1]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    if not HAS_TRITON or not q.is_cuda or D > 128:
        return triton_block_sparse_streaming_attn_pytorch_ref(
            q, k, v, active_block_indices, block_size=block_size, sm_scale=sm_scale
        )

    out = torch.empty_like(v)

    grid = (
        B,
        H,
        triton.cdiv(L, block_size),
    )

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    ab_c = active_block_indices.contiguous().to(torch.int32)

    _block_sparse_streaming_attn_kernel[grid](
        q_c,
        k_c,
        v_c,
        ab_c,
        out,
        B,
        H,
        L,
        D=D,
        K_ACTIVE=K_ACTIVE,
        BLOCK_SIZE=block_size,
        BLOCK_M=block_size,
        BLOCK_N=block_size,
        sm_scale=sm_scale,
        stride_qb=q_c.stride(0),
        stride_qh=q_c.stride(1),
        stride_ql=q_c.stride(2),
        stride_qd=q_c.stride(3),
        stride_kb=k_c.stride(0),
        stride_kh=k_c.stride(1),
        stride_kl=k_c.stride(2),
        stride_kd=k_c.stride(3),
        stride_vb=v_c.stride(0),
        stride_vh=v_c.stride(1),
        stride_vl=v_c.stride(2),
        stride_vd=v_c.stride(3),
        stride_ab_b=ab_c.stride(0),
        stride_ab_h=ab_c.stride(1),
        stride_ab_q=ab_c.stride(2),
        stride_ab_k=ab_c.stride(3),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
        stride_ol=out.stride(2),
        stride_od=out.stride(3),
    )

    return out


def fused_dsa_sparse_attn_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int = 64,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Clean PyTorch reference for DSA block-sparse attention.

    Loads only selected K/V blocks into attention computation.

    Args:
        q: Query tensor of shape [B, L, D] or [B, H, L, d]
        k: Key tensor of shape [B, L, D] or [B, H, L, d]
        v: Value tensor of shape [B, L, D] or [B, H, L, d]
        block_mask: Boolean tensor of shape [B, num_blocks] or indices [B, top_k]
        block_size: Block size B (default 64)
        sm_scale: Optional softmax scale factor (default 1 / sqrt(d))

    Returns:
        Output tensor matching q shape
    """
    _orig_shape = q.shape
    is_3d = (q.ndim == 3)
    if is_3d:
        B, L, D = q.shape
        H = 8
        d = D // H
        q_4d = q.view(B, L, H, d).permute(0, 2, 1, 3).contiguous()
        k_4d = k.view(B, L, H, d).permute(0, 2, 1, 3).contiguous()
        v_4d = v.view(B, L, H, d).permute(0, 2, 1, 3).contiguous()
    else:
        B, H, L, d = q.shape
        D = H * d
        q_4d = q.contiguous()
        k_4d = k.contiguous()
        v_4d = v.contiguous()

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    num_blocks = (L + block_size - 1) // block_size

    # Normalize block_mask to active_block_indices [B, H, num_blocks, top_k]
    if block_mask.dtype == torch.bool:
        # block_mask: [B, num_blocks]
        # extract selected block indices per batch
        B_size = block_mask.shape[0]
        active_list = []
        for b in range(B_size):
            nonzero = torch.nonzero(block_mask[b], as_tuple=False).squeeze(-1)
            active_list.append(nonzero)
        max_k = max(len(idx) for idx in active_list)
        active_indices = torch.full((B_size, max_k), -1, dtype=torch.long, device=block_mask.device)
        for b in range(B_size):
            active_indices[b, :len(active_list[b])] = active_list[b]
    else:
        active_indices = block_mask

    top_k = active_indices.shape[-1]
    active_block_indices = active_indices.unsqueeze(1).unsqueeze(2).expand(B, H, num_blocks, top_k).contiguous()

    out_4d = triton_block_sparse_streaming_attn_pytorch_ref(
        q_4d, k_4d, v_4d, active_block_indices, block_size=block_size, sm_scale=sm_scale
    )

    if is_3d:
        return out_4d.permute(0, 2, 1, 3).contiguous().view(B, L, D)
    return out_4d


def fused_dsa_sparse_attn_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int = 64,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Fused DSA block-sparse attention using Triton SRAM streaming.

    For each query block, loads only selected K/V blocks into SRAM.

    Args:
        q: Query tensor of shape [B, L, D] or [B, H, L, d]
        k: Key tensor of shape [B, L, D] or [B, H, L, d]
        v: Value tensor of shape [B, L, D] or [B, H, L, d]
        block_mask: Boolean tensor of shape [B, num_blocks] or indices [B, top_k]
        block_size: Block size B (default 64)
        sm_scale: Optional softmax scale factor (default 1 / sqrt(d))

    Returns:
        Output tensor matching q shape
    """
    _orig_shape = q.shape
    is_3d = (q.ndim == 3)
    if is_3d:
        B, L, D = q.shape
        H = 8
        d = D // H
        q_4d = q.view(B, L, H, d).permute(0, 2, 1, 3).contiguous()
        k_4d = k.view(B, L, H, d).permute(0, 2, 1, 3).contiguous()
        v_4d = v.view(B, L, H, d).permute(0, 2, 1, 3).contiguous()
    else:
        B, H, L, d = q.shape
        D = H * d
        q_4d = q.contiguous()
        k_4d = k.contiguous()
        v_4d = v.contiguous()

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    if not HAS_TRITON or not q.is_cuda:
        return fused_dsa_sparse_attn_pytorch_ref(
            q, k, v, block_mask, block_size=block_size, sm_scale=sm_scale
        )

    num_blocks = (L + block_size - 1) // block_size

    # Convert block_mask to active_block_indices [B, H, num_blocks, top_k]
    if block_mask.dtype == torch.bool:
        B_size = block_mask.shape[0]
        active_list = []
        for b in range(B_size):
            nonzero = torch.nonzero(block_mask[b], as_tuple=False).squeeze(-1)
            active_list.append(nonzero)
        max_k = max(len(idx) for idx in active_list)
        active_indices = torch.full((B_size, max_k), -1, dtype=torch.long, device=block_mask.device)
        for b in range(B_size):
            active_indices[b, :len(active_list[b])] = active_list[b]
    else:
        active_indices = block_mask

    top_k = active_indices.shape[-1]
    active_block_indices = active_indices.unsqueeze(1).unsqueeze(2).expand(B, H, num_blocks, top_k).contiguous()

    try:
        from neuralops.core.triton_kernels.dynamic_sparse_attention import triton_dynamic_sparse_attention
        out_4d = triton_dynamic_sparse_attention(
            q_4d, k_4d, v_4d, active_block_indices, block_size=block_size, sm_scale=sm_scale
        )
    except Exception:
        out_4d = triton_block_sparse_streaming_attn(
            q_4d, k_4d, v_4d, active_block_indices, block_size=block_size, sm_scale=sm_scale
        )

    if is_3d:
        return out_4d.permute(0, 2, 1, 3).contiguous().view(B, L, D)
    return out_4d


__all__ = [
    "butterfly_mask",
    "block_sparse_attention",
    "fused_moba_block_router_combine_triton",
    "fused_moba_block_router_combine_pytorch",
    "triton_block_sparse_streaming_attn",
    "triton_block_sparse_streaming_attn_pytorch_ref",
    "fused_dsa_sparse_attn_triton",
    "fused_dsa_sparse_attn_pytorch_ref",
]

