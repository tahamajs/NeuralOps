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
        with open(curr_dir / "selective_ssm_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "selective_ssm.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_selective_ssm", cpp_src, cuda_src, ["selective_ssm_fwd"])
    return _MODULE

def selective_ssm_ref(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B_vec: torch.Tensor, C_vec: torch.Tensor) -> torch.Tensor:
    B, N, D = x.shape
    out = torch.empty_like(x)
    h = torch.zeros(B, D, device=x.device, dtype=torch.float32)
    a_exp = A.view(1, 1, D)
    x_f = x.float()
    B_f = B_vec.float()
    C_f = C_vec.float()
    for t in range(N):
        delta = dt[:, t, :].float().clamp_min(1e-6)
        dA = torch.exp(-torch.clamp(delta * torch.abs(a_exp.squeeze(1)), min=0.0))
        dB = delta * B_f[:, t, :]
        h = dA * h + dB * x_f[:, t, :]
        h = torch.clamp(h, -65504.0, 65504.0)
        out[:, t, :] = (h * C_f[:, t, :]).to(x.dtype)
    return out

def selective_ssm(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B_vec: torch.Tensor, C_vec: torch.Tensor) -> torch.Tensor:
    def cuda_fn(x, dt, A, B_vec, C_vec):
        mod = _get_module()
        return mod.selective_ssm_fwd(x.contiguous(), dt.contiguous(), A.contiguous(), B_vec.contiguous(), C_vec.contiguous())
    return dispatch(cuda_fn, selective_ssm_ref, selective_ssm_ref, x, dt=dt, A=A, B_vec=B_vec, C_vec=C_vec, kernel_name="selective_ssm")
