"""
Unified Custom Op and Torch.compile Compatibility Registry for NeuralOps Triton Kernels.

Provides seamless support for `torch.compile(model, fullgraph=False)` without graph breaks:
  1. Uses PyTorch >= 2.4 `torch.library.custom_op` and `register_fake` when available.
  2. Provides automatic FakeTensor / Meta shape propagation.
  3. Falls back gracefully on older PyTorch versions or non-CUDA environments.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple
import torch
import torch.nn.functional as F

from neuralops.core.triton_kernels._common import (
    linear_attn_pytorch_fallback,
    sparse_attn_pytorch_fallback,
    low_rank_pytorch_fallback,
    state_update_pytorch_fallback,
    gqa_pytorch_fallback,
    triton_dispatch,
)

logger = logging.getLogger("NeuralOps.OpRegistry")

# Detect PyTorch 2.4+ modern torch.library custom_op support
_HAS_TORCH_LIBRARY_CUSTOM_OP = hasattr(torch.library, "custom_op") and hasattr(torch.library, "register_fake")


# ---------------------------------------------------------------------------
# Custom Op Registrations for torch.compile
# ---------------------------------------------------------------------------

if _HAS_TORCH_LIBRARY_CUSTOM_OP:
    try:
        # 1. Linear Attention Op
        @torch.library.custom_op("neuralops::linear_attn", mutates_args=())
        def _op_linear_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return triton_dispatch("linear_attn", q, k, v)

        @_op_linear_attn.register_fake
        def _op_linear_attn_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            # Output matches (B, H, N, D_v)
            return torch.empty_like(v)

        # 2. Sparse Attention Op
        @torch.library.custom_op("neuralops::sparse_attn", mutates_args=())
        def _op_sparse_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return triton_dispatch("sparse_attn", q, k, v)

        @_op_sparse_attn.register_fake
        def _op_sparse_attn_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(q)

        # 3. Low Rank Op
        @torch.library.custom_op("neuralops::low_rank", mutates_args=())
        def _op_low_rank(x: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
            return triton_dispatch("low_rank", x, U, V)

        @_op_low_rank.register_fake
        def _op_low_rank_fake(x: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
            # (..., in_dim) @ U (in_dim, rank) @ V (rank, out_dim) -> (..., out_dim)
            out_shape = list(x.shape[:-1]) + [V.shape[-1]]
            return x.new_empty(out_shape)

        # 4. GQA Op
        @torch.library.custom_op("neuralops::gqa", mutates_args=())
        def _op_gqa(k: torch.Tensor, v: torch.Tensor, num_kv_heads: int, num_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return triton_dispatch("gqa", k, v, num_kv_heads=num_kv_heads, num_heads=num_heads)

        @_op_gqa.register_fake
        def _op_gqa_fake(k: torch.Tensor, v: torch.Tensor, num_kv_heads: int, num_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
            ratio = num_heads // num_kv_heads
            k_out = k.repeat_interleave(ratio, dim=1) if k.dim() >= 3 and k.shape[1] == num_kv_heads else k
            v_out = v.repeat_interleave(ratio, dim=1) if v.dim() >= 3 and v.shape[1] == num_kv_heads else v
            return k_out, v_out

        _CUSTOM_OPS_REGISTERED = True
    except Exception as exc:
        logger.debug("Failed registering custom ops with torch.library: %s", exc)
        _CUSTOM_OPS_REGISTERED = False
else:
    _CUSTOM_OPS_REGISTERED = False


# ---------------------------------------------------------------------------
# High-Level Compilable Wrappers
# ---------------------------------------------------------------------------

def compilable_linear_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Torch.compile-friendly entrypoint for linear attention."""
    if _CUSTOM_OPS_REGISTERED and not kwargs:
        try:
            return torch.ops.neuralops.linear_attn(q, k, v)
        except Exception:
            pass
    return triton_dispatch("linear_attn", q, k, v, **kwargs)


def compilable_sparse_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Torch.compile-friendly entrypoint for sparse attention."""
    if _CUSTOM_OPS_REGISTERED and not kwargs:
        try:
            return torch.ops.neuralops.sparse_attn(q, k, v)
        except Exception:
            pass
    return triton_dispatch("sparse_attn", q, k, v, **kwargs)


def compilable_low_rank(x: torch.Tensor, U: torch.Tensor, V: torch.Tensor, bias: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
    """Torch.compile-friendly entrypoint for low rank factorization."""
    if _CUSTOM_OPS_REGISTERED and bias is None and not kwargs:
        try:
            return torch.ops.neuralops.low_rank(x, U, V)
        except Exception:
            pass
    return triton_dispatch("low_rank", x, U, V, bias=bias, **kwargs)


def register_custom_op_meta_shape(op_name: str, fake_fn: Callable) -> bool:
    """Register custom fake/meta implementation for torch.compile shape propagation."""
    if hasattr(torch.library, "register_fake"):
        try:
            torch.library.register_fake(op_name, fake_fn)
            return True
        except Exception as e:
            logger.debug("register_fake failed for %s: %s", op_name, e)
    return False


__all__ = [
    "compilable_linear_attn",
    "compilable_sparse_attn",
    "compilable_low_rank",
    "register_custom_op_meta_shape",
]
