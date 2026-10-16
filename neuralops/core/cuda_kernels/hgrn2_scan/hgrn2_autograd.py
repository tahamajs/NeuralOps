"""
HGRN2 Bidirectional Scan Interface and 3-Level Fallback.
Target: SM 8.9 (NVIDIA Ada Lovelace RTX 4070 Laptop GPU, 8GB VRAM).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "hgrn2_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "hgrn2_scan.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_hgrn2_scan",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["hgrn2_bidirectional_scan_fwd"],
        )
    return _MODULE


def hgrn2_fused_bwd_scan_pytorch_ref(
    x: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    from neuralops.core.triton_kernels.hgrn2_scan import triton_hgrn2_fused_bwd_scan_pytorch_ref
    return triton_hgrn2_fused_bwd_scan_pytorch_ref(x, g)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.hgrn2_scan import triton_hgrn2_fused_bwd_scan
        return triton_hgrn2_fused_bwd_scan
    except ImportError:
        return None


def hgrn2_fused_bwd_scan(
    x: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    """3-Level Dispatched Bidirectional HGRN2 Scan."""
    def cuda_fn(x, g):
        mod = _get_module()
        return mod.hgrn2_bidirectional_scan_fwd(x.contiguous(), g.contiguous())

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn=cuda_fn,
        triton_fn=triton_fn,
        pytorch_fn=hgrn2_fused_bwd_scan_pytorch_ref,
        args=(x, g),
        op_name="hgrn2_fused_bwd_scan",
    )
