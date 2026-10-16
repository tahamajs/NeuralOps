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
        with open(curr_dir / "vspec_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "vspec.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_vspec", cpp_src, cuda_src, ["vspec_fwd"])
    return _MODULE

def vspec_ref(x: torch.Tensor, spec_gate: torch.Tensor, decay: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, N, D = x.shape
    g = torch.sigmoid(spec_gate)
    out = torch.empty_like(x)
    s = torch.zeros(B, D, device=x.device, dtype=x.dtype)
    for t in range(N):
        s = decay[:, t, :] * s + (x[:, t, :] * g[:, t, :])
        out[:, t, :] = s * torch.rsqrt(s.pow(2) + eps)
    return out

def vspec(x: torch.Tensor, spec_gate: torch.Tensor, decay: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    def cuda_fn(x, spec_gate, decay, eps=1e-6):
        mod = _get_module()
        return mod.vspec_fwd(x.contiguous(), spec_gate.contiguous(), decay.contiguous(), float(eps))
    return dispatch(cuda_fn, vspec_ref, vspec_ref, x, spec_gate=spec_gate, decay=decay, eps=eps, kernel_name="vspec")
