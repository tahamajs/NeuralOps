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
        with open(curr_dir / "orion_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "orion.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_orion", cpp_src, cuda_src, ["orion_fwd"])
    return _MODULE

def orion_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    phi_q = torch.exp(-0.5 * q.float().pow(2) * temperature)
    phi_k = torch.exp(-0.5 * k.float().pow(2) * temperature)
    B, H, N, D = q.shape
    out = torch.empty_like(q)
    s = torch.zeros(B, H, D, device=q.device, dtype=torch.float32)
    v_f = v.float()
    for t in range(N):
        s = s + phi_k[:, :, t, :] * v_f[:, :, t, :]
        out[:, :, t, :] = (phi_q[:, :, t, :] * s).to(q.dtype)
    return out

def orion(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    def cuda_fn(q, k, v, temperature=1.0):
        mod = _get_module()
        return mod.orion_fwd(q.contiguous(), k.contiguous(), v.contiguous(), float(temperature))
    return dispatch(cuda_fn, orion_ref, orion_ref, q, k=k, v=v, temperature=temperature, kernel_name="orion")
