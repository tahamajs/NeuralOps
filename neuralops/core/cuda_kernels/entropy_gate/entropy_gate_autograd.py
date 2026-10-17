"""
Spectral Entropy SVD Dynamic Gate Interface and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "entropy_gate_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "entropy_gate.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_entropy_gate",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["spectral_entropy_svd_gate_fwd"],
        )
    return _MODULE


def spectral_entropy_svd_gate_pytorch_ref(
    x: torch.Tensor,
    V_r: torch.Tensor,
    U_r: torch.Tensor,
    effective_rank: int,
    s_weights: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    from neuralops.core.triton_kernels.low_rank import triton_spectral_entropy_svd_gate_pytorch_ref
    return triton_spectral_entropy_svd_gate_pytorch_ref(
        x, V_r, U_r, effective_rank, s_weights=s_weights, bias=bias
    )


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.low_rank import triton_spectral_entropy_svd_gate
        return triton_spectral_entropy_svd_gate
    except ImportError:
        return None


def spectral_entropy_svd_gate(
    x: torch.Tensor,
    V_r: torch.Tensor,
    U_r: torch.Tensor,
    effective_rank: int,
    s_weights: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """3-Level Dispatched Spectral Entropy SVD Dynamic Gate."""
    def cuda_fn(x, V_r, U_r, effective_rank, s_weights=None, bias=None, **kwargs):
        mod = _get_module()
        return mod.spectral_entropy_svd_gate_fwd(
            x.contiguous(),
            V_r.contiguous(),
            U_r.contiguous(),
            int(effective_rank),
            s_weights.contiguous() if s_weights is not None else None,
            bias.contiguous() if bias is not None else None,
        )

    triton_fn = _get_triton_impl()
    r_max = V_r.shape[1] if V_r.shape[0] == x.shape[-1] else V_r.shape[0]
    eff_r = min(int(effective_rank), r_max)
    if eff_r > 128:
        cuda_fn = None

    return dispatch(
        cuda_fn,
        triton_fn,
        spectral_entropy_svd_gate_pytorch_ref,
        x,
        V_r,
        U_r,
        effective_rank,
        s_weights=s_weights,
        bias=bias,
        kernel_name="spectral_entropy_svd_gate",
    )
