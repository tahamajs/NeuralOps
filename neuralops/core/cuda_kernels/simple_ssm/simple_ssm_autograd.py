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
        with open(curr_dir / "simple_ssm_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "simple_ssm.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_simple_ssm", cpp_src, cuda_src, ["simple_ssm_fwd"])
    return _MODULE

def simple_ssm_ref(x: torch.Tensor, decay: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    B, N, D = x.shape
    out = torch.empty_like(x)
    s = torch.zeros(B, D, device=x.device, dtype=x.dtype)
    for t in range(N):
        s = decay[:, t, :] * s + x[:, t, :]
        out[:, t, :] = s * scale
    return out

def simple_ssm(x: torch.Tensor, decay: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    def cuda_fn(x, decay, scale):
        mod = _get_module()
        return mod.simple_ssm_fwd(x.contiguous(), decay.contiguous(), scale.contiguous())
    return dispatch(cuda_fn, simple_ssm_ref, simple_ssm_ref, x, decay=decay, scale=scale, kernel_name="simple_ssm")
