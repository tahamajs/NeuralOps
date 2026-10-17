"""
Fused Dropout + Residual Addition Interface and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "dropout_add_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "fused_dropout_add.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_fused_dropout_add",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["fused_dropout_add_fwd"],
        )
    return _MODULE


def fused_dropout_add_pytorch_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    p: float = 0.0,
    training: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    dropped = F.dropout(x, p=p, training=training)
    return dropped + residual


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.elementwise import dropout_add_triton
        return dropout_add_triton
    except (ImportError, AttributeError):
        return None


def fused_dropout_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    p: float = 0.0,
    training: bool = False,
    seed: int | None = None,
) -> torch.Tensor:
    """3-Level Dispatched Fused Dropout + Add."""
    if seed is None:
        seed = int(torch.empty((), dtype=torch.int64).random_(0, 2**31).item())

    def cuda_fn(x, residual, p=0.0, training=False, seed=0, **kwargs):
        mod = _get_module()
        return mod.fused_dropout_add_fwd(
            x.contiguous(),
            residual.contiguous(),
            float(p),
            bool(training),
            int(seed),
        )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        fused_dropout_add_pytorch_ref,
        x,
        residual,
        p=p,
        training=training,
        seed=seed,
        kernel_name="fused_dropout_add",
    )
