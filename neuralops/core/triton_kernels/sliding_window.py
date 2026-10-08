"""Real Triton kernel for Sliding Window Attention with CPU/CUDA fallback."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels._common import HAS_TRITON, triton, tl

__all__ = ["sliding_window_attention_kernel", "sliding_window_triton"]


if HAS_TRITON and tl is not None:
    @triton.jit
    def sliding_window_attn_fwd_kernel(
        Q, K, V, Out,
        stride_qb, stride_qh, stride_qn, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_on, stride_od,
        H, N, WINDOW: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        i_b = pid_bh // H
        i_h = pid_bh % H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        q_ptrs = Q + i_b * stride_qb + i_h * stride_qh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)

        # Online softmax state
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # Iterate over key blocks within window
        start_col = tl.maximum(0, (pid_m * BLOCK_M) - WINDOW)
        end_col = tl.minimum(N, (pid_m + 1) * BLOCK_M)
        num_k_blocks = tl.cdiv(end_col - start_col, BLOCK_N)

        for j in range(num_k_blocks):
            offs_n = start_col + j * BLOCK_N + tl.arange(0, BLOCK_N)
            k_ptrs = K + i_b * stride_kb + i_h * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
            k = tl.load(k_ptrs, mask=offs_n[None, :] < N, other=0.0)

            scores = tl.dot(q, k) * (1.0 / tl.sqrt(tl.cast(BLOCK_D, tl.float32)))
            # Mask out outside window: (offs_m[:, None] - offs_n[None, :] > WINDOW) | (offs_n[None, :] > offs_m[:, None])
            dist = offs_m[:, None] - offs_n[None, :]
            mask_window = (dist >= 0) & (dist < WINDOW) & (offs_m[:, None] < N) & (offs_n[None, :] < N)
            scores = tl.where(mask_window, scores, float("-inf"))

            m_curr = tl.maximum(m_i, tl.max(scores, 1))
            alpha = tl.exp(m_i - m_curr)
            p = tl.exp(scores - m_curr[:, None])

            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            v_ptrs = V + i_b * stride_vb + i_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)
            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_curr

        out_ptrs = Out + i_b * stride_ob + i_h * stride_oh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
        out = acc / tl.maximum(l_i[:, None], 1e-6)
        tl.store(out_ptrs, out, mask=offs_m[:, None] < N)


def sliding_window_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 256,
) -> torch.Tensor:
    """Reference sliding window attention using PyTorch masked operations."""
    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Causal sliding window mask: (i - j >= 0) and (i - j < window_size)
    q_idx = torch.arange(N, device=q.device).unsqueeze(1)
    k_idx = torch.arange(N, device=q.device).unsqueeze(0)
    dist = q_idx - k_idx
    mask = (dist >= 0) & (dist < window_size)
    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    return torch.matmul(weights, v)


def sliding_window_attention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 256,
) -> torch.Tensor:
    """Dispatches sliding window attention to Triton GPU kernel if available, else PyTorch."""
    if not HAS_TRITON or q.device.type != "cuda":
        return sliding_window_pytorch(q, k, v, window_size=window_size)

    B, H, N, D = q.shape
    out = torch.empty_like(q)
    grid = (triton.cdiv(N, 64), B * H)
    sliding_window_attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H, N, WINDOW=window_size,
        BLOCK_M=64, BLOCK_N=64, BLOCK_D=D,
    )
    return out


sliding_window_triton = sliding_window_attention_kernel
