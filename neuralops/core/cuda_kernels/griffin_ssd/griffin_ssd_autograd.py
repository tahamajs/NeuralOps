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
        with open(curr_dir / "griffin_ssd_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "griffin_ssd.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_griffin_ssd", cpp_src, cuda_src, ["griffin_ssd_fwd"])
    return _MODULE

def griffin_ssd_ref(x: torch.Tensor, r: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    B, N, D = x.shape
    r_gate = torch.sigmoid(r)
    out = torch.empty_like(x)
    s = torch.zeros(B, D, device=x.device, dtype=x.dtype)
    for t in range(N):
        lam = decay[:, t, :]
        s = lam * s + (1.0 - lam) * x[:, t, :]
        out[:, t, :] = s * r_gate[:, t, :]
    return out

def griffin_ssd(x: torch.Tensor, r: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    def cuda_fn(x, r, decay):
        mod = _get_module()
        return mod.griffin_ssd_fwd(x.contiguous(), r.contiguous(), decay.contiguous())
    return dispatch(cuda_fn, griffin_ssd_ref, griffin_ssd_ref, x, r=r, decay=decay, kernel_name="griffin_ssd")
