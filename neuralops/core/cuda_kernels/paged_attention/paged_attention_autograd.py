"""
PagedAttention Autograd Wrapper & Fallback Interface (Module 1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "paged_attention_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "paged_attention.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_paged_attention",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["paged_attention_v1"],
        )
    return _MODULE


def paged_attention_pytorch_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_context_len: int,
    scale: float,
) -> torch.Tensor:
    """
    Unpaged PyTorch reference for numerical ground truth verification.
    """
    num_seqs, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    group_size = num_heads // num_kv_heads
    out = torch.zeros_like(q)

    for seq_idx in range(num_seqs):
        ctx_len = int(context_lens[seq_idx].item())
        if ctx_len <= 0:
            continue

        num_blocks = (ctx_len + block_size - 1) // block_size
        k_blocks = []
        v_blocks = []
        for b in range(num_blocks):
            p_block = int(block_tables[seq_idx, b].item())
            # [num_kv_heads, head_dim, block_size] -> [num_kv_heads, block_size, head_dim]
            k_t = k_cache[p_block].permute(0, 2, 1)
            v_t = v_cache[p_block].permute(0, 2, 1)
            k_blocks.append(k_t)
            v_blocks.append(v_t)

        k_all = torch.cat(k_blocks, dim=1)[:, :ctx_len, :]  # [num_kv_heads, ctx_len, head_dim]
        v_all = torch.cat(v_blocks, dim=1)[:, :ctx_len, :]

        # Repeat KV heads if GQA
        if group_size > 1:
            k_all = k_all.repeat_interleave(group_size, dim=0)  # [num_heads, ctx_len, head_dim]
            v_all = v_all.repeat_interleave(group_size, dim=0)

        q_seq = q[seq_idx].unsqueeze(1)  # [num_heads, 1, head_dim]
        scores = torch.matmul(q_seq, k_all.transpose(-1, -2)) * scale  # [num_heads, 1, ctx_len]
        probs = F.softmax(scores, dim=-1)
        o_seq = torch.matmul(probs, v_all).squeeze(1)  # [num_heads, head_dim]
        out[seq_idx] = o_seq

    return out


def paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int = 16,
    max_context_len: int | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Executes PagedAttention with automatic 3-level fallback.
    """
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    if max_context_len is None:
        max_context_len = int(context_lens.max().item()) if context_lens.numel() > 0 else 0

    mod = _get_module()
    cuda_fn = mod.paged_attention_v1 if mod is not None else None

    # Check for existing Triton paged attention kernel if available
    triton_fn = None
    try:
        from neuralops.core.triton_kernels.paged_attention import paged_attention_triton
        triton_fn = paged_attention_triton
    except Exception:
        pass

    return dispatch(
        cuda_fn,
        triton_fn,
        paged_attention_pytorch_reference,
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        scale,
        kernel_name="paged_attention",
    )
