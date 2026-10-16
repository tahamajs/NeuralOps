"""
Gated DeltaNet Chunked Scan Interface and 3-Level Fallback.
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
        with open(curr_dir / "gdn_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "chunked_delta_gate_scan.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_gdn_scan",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["chunked_delta_gate_scan_fwd"],
        )
    return _MODULE


def chunked_delta_gate_scan_pytorch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch reference implementation."""
    from neuralops.core.triton_kernels.gdn_scan import triton_chunked_delta_gate_scan_pytorch_ref
    return triton_chunked_delta_gate_scan_pytorch_ref(q, k, v, alpha, beta)


def _get_triton_impl():
    try:
        from neuralops.core.triton_kernels.gdn_scan import triton_chunked_delta_gate_scan
        return triton_chunked_delta_gate_scan
    except ImportError:
        return None


def chunked_delta_gate_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """3-Level Dispatched Chunked Delta-Gate Recurrent Scan."""
    def cuda_fn(q, k, v, alpha, beta):
        mod = _get_module()
        return mod.chunked_delta_gate_scan_fwd(
            q.contiguous(), k.contiguous(), v.contiguous(), alpha.contiguous(), beta.contiguous()
        )

    triton_fn = _get_triton_impl()

    return dispatch(
        cuda_fn=cuda_fn,
        triton_fn=triton_fn,
        pytorch_fn=chunked_delta_gate_scan_pytorch_ref,
        args=(q, k, v, alpha, beta),
        op_name="chunked_delta_gate_scan",
    )
