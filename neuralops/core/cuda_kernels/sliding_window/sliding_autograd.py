"""
Sliding Window Attention Python Interface & 3-Level Fallback (Module 7).
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
        with open(curr_dir / "sliding_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "sliding_window.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_sliding_window",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["sliding_window_attn_fwd"],
        )
    return _MODULE


def sliding_window_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 128,
    num_sinks: int = 4,
    scale: float | None = None,
) -> torch.Tensor:
    B, H, N, D = q.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    q_idx = torch.arange(N, device=q.device).unsqueeze(1)
    k_idx = torch.arange(N, device=q.device).unsqueeze(0)

    causal = k_idx <= q_idx
    in_window = k_idx >= (q_idx - window_size + 1)
    is_sink = k_idx < num_sinks

    valid_mask = causal & (in_window | is_sink)
    scores = scores.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    probs = F.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.matmul(probs, v)


def sliding_window_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int = 128,
    num_sinks: int = 4,
    scale: float | None = None,
) -> torch.Tensor:
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    mod = _get_module()
    cuda_fn = mod.sliding_window_attn_fwd if mod is not None else None

    triton_fn = None
    try:
        from neuralops.core.triton_kernels.sliding_window import sliding_window_attention_triton
    except Exception:
        pass

    return dispatch(
        cuda_fn,
        None,
        sliding_window_pytorch_ref,
        q,
        k,
        v,
        window_size,
        num_sinks,
        scale,
        kernel_name="sliding_window_attn",
    )
