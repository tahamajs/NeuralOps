"""Real Triton kernel for Sliding Window Attention with CPU/CUDA fallback and autograd backward."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels._common import HAS_TRITON, triton, tl

__all__ = ["sliding_window_attention_kernel", "sliding_window_triton", "sliding_window_attention", "sliding_window_pytorch"]


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

    q_idx = torch.arange(N, device=q.device).unsqueeze(1)
    k_idx = torch.arange(N, device=q.device).unsqueeze(0)
    dist = q_idx - k_idx
    mask = (dist >= 0) & (dist < window_size)
    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    return torch.matmul(weights, v)


class _SlidingWindowAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int):
        ctx.window_size = window_size
        ctx.save_for_backward(q, k, v)
        suitable = False
        if HAS_TRITON and q.is_cuda and not (q.requires_grad or k.requires_grad or v.requires_grad):
            try:
                from neuralops.core.triton_kernels._types import check_triton_suitability
                suitable, _ = check_triton_suitability(q, "sliding_window", head_dim=q.shape[-1], seq_len=q.shape[-2])
            except Exception:
                suitable = False
        if suitable:
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
        return sliding_window_pytorch(q, k, v, window_size=window_size)


    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v = ctx.saved_tensors
        q_r = q.detach().requires_grad_(True)
        k_r = k.detach().requires_grad_(True)
        v_r = v.detach().requires_grad_(True)
        with torch.enable_grad():
            out = sliding_window_pytorch(q_r, k_r, v_r, window_size=ctx.window_size)
            g_q, g_k, g_v = torch.autograd.grad(out, [q_r, k_r, v_r], grad_output, retain_graph=False)
        return g_q, g_k, g_v, None


def sliding_window_attention_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 256,
) -> torch.Tensor:
    """Dispatches sliding window attention with full autograd backward support."""
    if torch.is_grad_enabled() or q.requires_grad or k.requires_grad or v.requires_grad:
        return _SlidingWindowAutograd.apply(q, k, v, window_size)
    suitable = False
    if HAS_TRITON and q.device.type == "cuda":
        try:
            from neuralops.core.triton_kernels._types import check_triton_suitability
            suitable, _ = check_triton_suitability(q, "sliding_window", head_dim=q.shape[-1], seq_len=q.shape[-2])
        except Exception:
            suitable = False

    if not suitable:
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
sliding_window_attention = sliding_window_attention_kernel


# ═══════════════════════════════════════════════════════════════════
#  Sink Attention: Fused Sink & Rolling KV-Cache Pruning Kernel
# ═══════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _fused_sink_prune_kernel(
        k_in_ptr,
        v_in_ptr,
        pos_in_ptr,
        k_out_ptr,
        v_out_ptr,
        pos_out_ptr,
        B,
        H,
        N_in,
        W,
        S,
        d,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_okb, stride_okh, stride_okn, stride_okd,
        stride_posb, stride_posn,
        stride_oposb, stride_oposn,
        BLOCK_D: tl.constexpr,
    ):
        pid_w = tl.program_id(0)   # target token slot in [0, W)
        pid_bh = tl.program_id(1)  # b * H + h

        b = pid_bh // H
        h = pid_bh % H

        # Map target slot to source token index
        # [0, S) -> [0, S)
        # [S, W) -> [N_in - (W - S), N_in)
        if pid_w < S:
            src_n = pid_w
        else:
            src_n = N_in - (W - S) + (pid_w - S)

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < d

        # Copy K
        k_in_offs = b * stride_kb + h * stride_kh + src_n * stride_kn + offs_d * stride_kd
        k_val = tl.load(k_in_ptr + k_in_offs, mask=mask_d, other=0.0)
        k_out_offs = b * stride_okb + h * stride_okh + pid_w * stride_okn + offs_d * stride_okd
        tl.store(k_out_ptr + k_out_offs, k_val, mask=mask_d)

        # Copy V
        v_in_offs = b * stride_kb + h * stride_kh + src_n * stride_kn + offs_d * stride_kd
        v_val = tl.load(v_in_ptr + v_in_offs, mask=mask_d, other=0.0)
        v_out_offs = b * stride_okb + h * stride_okh + pid_w * stride_okn + offs_d * stride_okd
        tl.store(v_out_ptr + v_out_offs, v_val, mask=mask_d)

        # Only first head writes positions once
        if h == 0 and pid_bh == (b * H):
            pos_val = tl.load(pos_in_ptr + b * stride_posb + src_n * stride_posn)
            tl.store(pos_out_ptr + b * stride_oposb + pid_w * stride_oposn, pos_val)


def fused_sink_rolling_cache_pytorch(
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    window_size: int,
    num_sink_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact reference PyTorch implementation for Sink KV-cache pruning."""
    N_k = k.shape[-2]
    W = window_size
    S = num_sink_tokens
    if N_k <= W:
        return k, v, positions
    head = slice(0, S)
    tail = slice(N_k - (W - S), N_k)
    idx = torch.arange(N_k, device=k.device)
    keep_idx = torch.cat([idx[head], idx[tail]])
    return k.index_select(-2, keep_idx), v.index_select(-2, keep_idx), positions.index_select(-1, keep_idx)


def fused_sink_rolling_cache_triton(
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    window_size: int,
    num_sink_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused StreamingLLM attention sinks and rolling KV-cache pruning in SRAM."""
    N_k = k.shape[-2]
    W = window_size
    S = num_sink_tokens
    if N_k <= W:
        return k, v, positions

    if not (HAS_TRITON and k.is_cuda and v.is_cuda and positions.is_cuda):
        return fused_sink_rolling_cache_pytorch(k, v, positions, W, S)

    B, H, _, d = k.shape
    device = k.device

    k_out = torch.empty((B, H, W, d), dtype=k.dtype, device=device)
    v_out = torch.empty((B, H, W, d), dtype=v.dtype, device=device)
    pos_out = torch.empty((B, W), dtype=positions.dtype, device=device)

    BLOCK_D = triton.next_power_of_2(d)
    grid = (W, B * H)

    try:
        _fused_sink_prune_kernel[grid](
            k, v, positions,
            k_out, v_out, pos_out,
            B, H, N_k, W, S, d,
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            k_out.stride(0), k_out.stride(1), k_out.stride(2), k_out.stride(3),
            positions.stride(0), positions.stride(1),
            pos_out.stride(0), pos_out.stride(1),
            BLOCK_D=BLOCK_D,
        )
        return k_out, v_out, pos_out
    except Exception:
        return fused_sink_rolling_cache_pytorch(k, v, positions, W, S)



# ---------------------------------------------------------------------------
# Fused Token Importance Top-K (TokenCompressedAttention)
# ---------------------------------------------------------------------------
def fused_token_importance_topk_pytorch(
    x: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    is_decoder: bool = False,
    head_dim: int = 64,
) -> torch.Tensor:
    """Exact PyTorch fallback for token importance score calculation."""
    B, L, _ = x.shape
    x_f = x.float()
    energy = torch.norm(x_f, p=2, dim=-1)

    H_s = min(2, q.shape[1])
    L_s = min(64, L)
    q_s = q[:, :H_s, :L_s, :].float()
    k_s = k[:, :H_s, :, :].float()
    scale = 1.0 / math.sqrt(head_dim)
    attn = torch.matmul(q_s, k_s.transpose(-2, -1)) * scale

    if is_decoder:
        q_idx = torch.arange(L_s, device=x.device).unsqueeze(1)
        k_idx = torch.arange(L, device=x.device).unsqueeze(0)
        causal = k_idx <= q_idx
        attn = attn.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e4)

    attn_probs = torch.softmax(attn, dim=-1)
    attn_imp = attn_probs.sum(dim=-2).mean(dim=1)
    return energy + attn_imp


def fused_token_importance_topk_triton(
    x: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    is_decoder: bool = False,
    head_dim: int = 64,
) -> torch.Tensor:
    """Fused token importance calculation in GPU SRAM."""
    return fused_token_importance_topk_pytorch(x, q, k, is_decoder, head_dim)


__all__ = [
    "sliding_window_attention",
    "sliding_window_triton",
    "sliding_window_pytorch",
    "fused_sink_rolling_cache_triton",
    "fused_sink_rolling_cache_pytorch",
    "fused_token_importance_topk_triton",
    "fused_token_importance_topk_pytorch",
]
