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
        with open(curr_dir / "aegis_gate_norm_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "aegis_gate_norm.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module(
            name="cuda_aegis_gate_norm",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["aegis_gate_norm_2branch_fwd"],
        )
    return _MODULE

def aegis_gate_norm_pytorch_ref(b0, b1, weights, eps=1e-5):
    from neuralops.core.triton_kernels.elementwise import fused_aegis_multibranch_gate_norm_pytorch
    return fused_aegis_multibranch_gate_norm_pytorch([b0, b1], weights, eps=eps)
def aegis_gate_norm_triton_ref(b0, b1, weights, eps=1e-5):
    from neuralops.core.triton_kernels.elementwise import fused_aegis_multibranch_gate_norm_triton
    return fused_aegis_multibranch_gate_norm_triton([b0, b1], weights, eps=eps)

def aegis_gate_norm(b0: torch.Tensor, b1: torch.Tensor, weights: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    def cuda_fn(*args, **kwargs):
        mod = _get_module()
        fn = getattr(mod, "aegis_gate_norm_2branch_fwd")
        return fn(*args, **kwargs)
    return dispatch(
        cuda_fn,
        aegis_gate_norm_triton_ref,
        aegis_gate_norm_pytorch_ref,
        b0, b1, weights, eps=eps,
        kernel_name="aegis_gate_norm",
    )
