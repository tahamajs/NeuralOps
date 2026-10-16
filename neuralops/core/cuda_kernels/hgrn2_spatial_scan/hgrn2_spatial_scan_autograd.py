from __future__ import annotations
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None

def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "hgrn2_spatial_scan_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "hgrn2_spatial_scan.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_hgrn2_spatial_scan",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["hgrn2_spatial_scan_fwd"],
        )
    return _MODULE

def hgrn2_spatial_scan_pytorch_ref(q, k, v, g):
    from neuralops.core.triton_kernels.hgrn2_scan import hgrn2_scan_pytorch
    y, _ = hgrn2_scan_pytorch(q, k, v, g)
    return y
def hgrn2_spatial_scan_triton_ref(q, k, v, g):
    from neuralops.core.triton_kernels.hgrn2_scan import hgrn2_scan_triton
    return hgrn2_scan_triton(q, k, v, g)

def hgrn2_spatial_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    def cuda_fn(*args, **kwargs):
        mod = _get_module()
        fn = getattr(mod, "hgrn2_spatial_scan_fwd")
        return fn(*args, **kwargs)
    return dispatch(
        cuda_fn,
        hgrn2_spatial_scan_triton_ref,
        hgrn2_spatial_scan_pytorch_ref,
        q, k, v, g,
        kernel_name="hgrn2_spatial_scan",
    )
