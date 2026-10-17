from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
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
    B, H, N, D = q.shape
    phi_q = F.elu(q / float(temperature)) + 1.0
    phi_k = F.elu(k / float(temperature)) + 1.0
    out = torch.empty_like(v)
    s = torch.zeros(B, H, D, device=q.device, dtype=torch.float32)
    phi_k_f = phi_k.float()
    v_f = v.float()
    phi_q_f = phi_q.float()
    for t in range(N):
        s = s + phi_k_f[:, :, t, :] * v_f[:, :, t, :]
        out[:, :, t, :] = (phi_q_f[:, :, t, :] * s).to(q.dtype)
    return out

def _orion_fwd_dispatch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    def cuda_fn(q, k, v, temperature=1.0, **kwargs):
        mod = _get_module()
        return mod.orion_fwd(q.contiguous(), k.contiguous(), v.contiguous(), float(temperature))
    return dispatch(cuda_fn, orion_ref, orion_ref, q, k=k, v=v, temperature=temperature, kernel_name="orion")

class _OrionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q, k, v, temperature: float = 1.0):
        out = _orion_fwd_dispatch(q, k, v, temperature)
        ctx.save_for_backward(q, k, v)
        ctx.temperature = float(temperature)
        return out

    @staticmethod
    def backward(ctx: Any, dout):
        q, k, v = ctx.saved_tensors
        q_p = q.detach().requires_grad_(q.requires_grad)
        k_p = k.detach().requires_grad_(k.requires_grad)
        v_p = v.detach().requires_grad_(v.requires_grad)

        with torch.enable_grad():
            out_p = orion_ref(q_p, k_p, v_p, ctx.temperature)

        grads = torch.autograd.grad(out_p, [q_p, k_p, v_p], dout, allow_unused=True)
        return grads[0], grads[1], grads[2], None

def orion(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if any(t.requires_grad for t in (q, k, v) if isinstance(t, torch.Tensor)):
        return _OrionFunction.apply(q, k, v, temperature)
    return _orion_fwd_dispatch(q, k, v, temperature)
