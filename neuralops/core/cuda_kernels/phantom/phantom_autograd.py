from __future__ import annotations
from pathlib import Path
import torch
from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None
def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "phantom_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "phantom.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_phantom", cpp_src, cuda_src, ["phantom_fwd"])
    return _MODULE

def phantom_ref(x: torch.Tensor, gate: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Pure PyTorch reference implementation with FP32 precision for threshold boundary."""
    g = torch.sigmoid(gate.float())
    mask = torch.where(g >= threshold, g, torch.zeros_like(g)).to(x.dtype)
    return x * mask

def phantom(x: torch.Tensor, gate: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    def cuda_fn(x, gate, threshold=0.5, **kwargs):
        mod = _get_module()
        return mod.phantom_fwd(x.contiguous(), gate.contiguous(), float(threshold))
    return dispatch(cuda_fn, phantom_ref, phantom_ref, x, gate, threshold=threshold, kernel_name="phantom")
