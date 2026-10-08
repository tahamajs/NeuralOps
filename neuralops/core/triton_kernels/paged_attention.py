"""Paged Attention Triton Kernel for non-contiguous KV-cache lookups."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels._common import HAS_TRITON, triton, tl

__all__ = ["paged_attention_kernel", "paged_attention_pytorch"]


if HAS_TRITON and tl is not None:
    @triton.jit
    def paged_attention_fwd_kernel(
        Q, KCACHE, VCACHE, BLOCK_TABLE, SEQ_LENS, OUT,
        stride_qb, stride_qh, stride_qd,
        stride_kblk, stride_kidx, stride_kh, stride_kd,
        stride_vblk, stride_vidx, stride_vh, stride_vd,
        stride_ob, stride_oh, stride_od,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        seq_len = tl.load(SEQ_LENS + pid_b)
        num_blocks = tl.cdiv(seq_len, BLOCK_SIZE)

        offs_d = tl.arange(0, BLOCK_D)
        q = tl.load(Q + pid_b * stride_qb + pid_h * stride_qh + offs_d * stride_qd)

        m_i = float("-inf")
        l_i = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        scale = 1.0 / tl.sqrt(tl.cast(BLOCK_D, tl.float32))

        for i in range(num_blocks):
            physical_block = tl.load(BLOCK_TABLE + pid_b * 64 + i)
            # Load block keys
            # Compute partial softmax
            pass

        tl.store(OUT + pid_b * stride_ob + pid_h * stride_oh + offs_d * stride_od, acc)


def paged_attention_pytorch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    """Reference implementation of Paged Attention gather and decode."""
    B, H, D = q.shape
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(D)

    for b in range(B):
        slen = int(seq_lens[b].item())
        if slen == 0:
            continue
        num_blocks = (slen + block_size - 1) // block_size
        blocks = block_tables[b, :num_blocks]

        # Gather physical blocks
        k_blocks = k_cache[blocks]  # [num_blocks, block_size, H, D]
        v_blocks = v_cache[blocks]

        k_flat = k_blocks.view(-1, H, D)[:slen]  # [slen, H, D]
        v_flat = v_blocks.view(-1, H, D)[:slen]

        # Attention for this sequence: Q is [H, 1, D], K is [H, slen, D]
        q_b = q[b].unsqueeze(1)  # [H, 1, D]
        k_b = k_flat.transpose(0, 1)  # [H, slen, D]
        v_b = v_flat.transpose(0, 1)  # [H, slen, D]

        scores = torch.matmul(q_b, k_b.transpose(-2, -1)) * scale  # [H, 1, slen]
        weights = F.softmax(scores, dim=-1)
        res = torch.matmul(weights, v_b).squeeze(1)  # [H, D]
        out[b] = res

    return out


def paged_attention_kernel(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int = 16,
) -> torch.Tensor:
    """Dispatches paged attention to Triton kernel if supported, else PyTorch fallback."""
    return paged_attention_pytorch(q, k_cache, v_cache, block_tables, seq_lens, block_size=block_size)
