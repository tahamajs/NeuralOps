"""
Variable-Length Attention

Processes batched sequences of variable lengths using cumulative token offsets without padding overhead.

Author: Taha Majlesi
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


from neuralops.core.triton_kernels._common import HAS_TRITON, triton, tl


if HAS_TRITON and tl is not None:
    @triton.jit
    def _varlen_flash_attn_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        cu_seqlens_q_ptr, cu_seqlens_k_ptr,
        stride_qn, stride_qh, stride_qd,
        stride_kn, stride_kh, stride_kd,
        stride_vn, stride_vh, stride_vd,
        stride_on, stride_oh, stride_od,
        H,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused Variable-Length Ragged Attention Kernel in SRAM."""
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch_id = pid_bh // H
        head_id = pid_bh % H

        q_start = tl.load(cu_seqlens_q_ptr + batch_id)
        q_end = tl.load(cu_seqlens_q_ptr + batch_id + 1)
        k_start = tl.load(cu_seqlens_k_ptr + batch_id)
        k_end = tl.load(cu_seqlens_k_ptr + batch_id + 1)

        seq_len_q = q_end - q_start
        seq_len_k = k_end - k_start

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq_len_q
        offs_d = tl.arange(0, BLOCK_D)

        q_ptrs = Q_ptr + (q_start + offs_m[:, None]) * stride_qn + head_id * stride_qh + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        sm_scale = 1.0 / tl.sqrt(tl.cast(BLOCK_D, tl.float32))

        m_i = tl.full([BLOCK_M], -1e4, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        num_k_blocks = tl.cdiv(seq_len_k, BLOCK_N)
        for k_blk in range(num_k_blocks):
            offs_n = k_blk * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seq_len_k

            k_ptrs = K_ptr + (k_start + offs_n[None, :]) * stride_kn + head_id * stride_kh + offs_d[:, None] * stride_kd
            k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float32)

            scores = tl.dot(q, k) * sm_scale
            if IS_CAUSAL:
                scores = tl.where(offs_m[:, None] >= offs_n[None, :], scores, -1e4)

            scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, -1e4)

            m_curr = tl.maximum(m_i, tl.max(scores, 1))
            alpha = tl.exp(m_i - m_curr)
            p = tl.exp(scores - m_curr[:, None])

            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            v_ptrs = V_ptr + (k_start + offs_n[:, None]) * stride_vn + head_id * stride_vh + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_curr

        out = acc / tl.maximum(l_i[:, None], 1e-6)
        out_ptrs = Out_ptr + (q_start + offs_m[:, None]) * stride_on + head_id * stride_oh + offs_d[None, :] * stride_od
        tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None])


def _varlen_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    B = cu_seqlens_q.shape[0] - 1
    total_q, H, D = q.shape

    from neuralops.core.triton_kernels._types import HAS_TRITON, FORCE_PYTORCH_FALLBACK
    if HAS_TRITON and not FORCE_PYTORCH_FALLBACK and q.is_cuda and D in (16, 32, 64, 128, 256):
        try:
            out = torch.empty_like(q)
            max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
            BLOCK_M = 64
            BLOCK_N = 64
            grid = (triton.cdiv(max_seqlen_q, BLOCK_M), B * H)
            _varlen_flash_attn_fwd_kernel[grid](
                q, k, v, out,
                cu_seqlens_q.to(device=q.device, dtype=torch.int32),
                cu_seqlens_k.to(device=k.device, dtype=torch.int32),
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                H,
                IS_CAUSAL=causal,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_D=D,
            )
            return out
        except Exception:
            pass

    outputs = []
    for b in range(B):
        qs, qe = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
        ks, ke = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()
        
        q_b = q[qs:qe].transpose(0, 1).unsqueeze(0)  # [1, H, Nq, D]
        k_b = k[ks:ke].transpose(0, 1).unsqueeze(0)
        v_b = v[ks:ke].transpose(0, 1).unsqueeze(0)
        
        out_b = F.scaled_dot_product_attention(
            q_b, k_b, v_b, is_causal=causal,
        )
        outputs.append(out_b.squeeze(0).transpose(0, 1))
    
    return torch.cat(outputs, dim=0)


class _VarlenAttentionAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        causal: bool,
    ):
        ctx.save_for_backward(q, k, v, cu_seqlens_q, cu_seqlens_k)
        ctx.causal = causal
        return _varlen_attention_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=causal)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        q, k, v, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        q_r = q.detach().requires_grad_(True)
        k_r = k.detach().requires_grad_(True)
        v_r = v.detach().requires_grad_(True)
        with torch.enable_grad():
            out = _varlen_attention_fwd(q_r, k_r, v_r, cu_seqlens_q, cu_seqlens_k, causal=ctx.causal)
            g_q, g_k, g_v = torch.autograd.grad(out, [q_r, k_r, v_r], grad_output, retain_graph=False)
        return g_q, g_k, g_v, None, None, None


def varlen_attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    """PyTorch implementation for varlen attention with full autograd backward support."""
    if torch.is_grad_enabled() or q.requires_grad or k.requires_grad or v.requires_grad:
        return _VarlenAttentionAutograd.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, causal)
    return _varlen_attention_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=causal)


varlen_attention_triton = _varlen_attention_fwd

__all__ = [
    "varlen_attention_pytorch",
    "varlen_attention_triton",
    "_varlen_attention_fwd",
]
