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
        with open(curr_dir / "hstu_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "hstu_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_hstu_attention", cpp_src, cuda_src, ["hstu_fwd"])
    return _MODULE

def hstu_attention_ref(u: torch.Tensor, v: torch.Tensor, decay: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, N, D = u.shape
    out = torch.empty_like(u)
    s = torch.zeros(B, D, device=u.device, dtype=u.dtype)
    for t in range(N):
        s = decay[:, t, :] * s + v[:, t, :]
        norm = s.abs() + eps
        out[:, t, :] = u[:, t, :] * (s / norm)
    return out

def hstu_attention(u: torch.Tensor, v: torch.Tensor, decay: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    def cuda_fn(u, v, decay, eps=1e-6):
        mod = _get_module()
        return mod.hstu_fwd(u.contiguous(), v.contiguous(), decay.contiguous(), float(eps))
    return dispatch(cuda_fn, hstu_attention_ref, hstu_attention_ref, u, v=v, decay=decay, eps=eps, kernel_name="hstu_attention")
