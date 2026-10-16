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
        with open(curr_dir / "svd_prefix_scan_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "svd_prefix_scan.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_svd_prefix_scan",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["svd_prefix_scan_fwd"],
        )
    return _MODULE

def svd_prefix_scan_pytorch_ref(q, k, v, eps=1e-4):
    kv = torch.einsum('bhld,bhle->bhlde', k.float(), v.float())
    s = kv.cumsum(dim=2)
    return torch.einsum('bhld,bhlde->bhle', q.float(), s).to(q.dtype)
def svd_prefix_scan_triton_ref(q, k, v, eps=1e-4):
    return svd_prefix_scan_pytorch_ref(q, k, v, eps=eps)

def svd_prefix_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    def cuda_fn(*args, **kwargs):
        mod = _get_module()
        fn = getattr(mod, "svd_prefix_scan_fwd")
        return fn(*args, **kwargs)
    return dispatch(
        cuda_fn,
        svd_prefix_scan_triton_ref,
        svd_prefix_scan_pytorch_ref,
        q, k, v, eps=eps,
        kernel_name="svd_prefix_scan",
    )
