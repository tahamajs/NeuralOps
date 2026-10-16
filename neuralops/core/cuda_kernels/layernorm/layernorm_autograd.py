"""
Fused LayerNorm Interface and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "layernorm_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "layernorm.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_layernorm",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["layernorm_fwd"],
        )
    return _MODULE


def layernorm_pytorch_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    orig_dtype = x.dtype
    x_eff = x.float() + (residual.float() if residual is not None else 0.0)
    mean = x_eff.mean(-1, keepdim=True)
    var = ((x_eff - mean) ** 2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    norm = (x_eff - mean) * rstd
    out = norm * weight.float()
    if bias is not None:
        out = out + bias.float()
    return out.to(orig_dtype)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.elementwise import layernorm_triton
        return layernorm_triton
    except (ImportError, AttributeError):
        return None


def layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """3-Level Dispatched Fused LayerNorm."""
    def cuda_fn(x, weight, bias=None, residual=None, eps=1e-6, **kwargs):
        mod = _get_module()
        return mod.layernorm_fwd(
            x.contiguous(),
            weight.contiguous(),
            bias.contiguous() if bias is not None else None,
            residual.contiguous() if residual is not None else None,
            float(eps),
        )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn,
        triton_fn,
        layernorm_pytorch_ref,
        x,
        weight,
        bias=bias,
        residual=residual,
        eps=eps,
        kernel_name="layernorm",
    )
