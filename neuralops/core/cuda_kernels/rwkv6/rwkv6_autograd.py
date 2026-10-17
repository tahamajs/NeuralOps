from __future__ import annotations
from typing import Any
from pathlib import Path
import torch
from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None
def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "rwkv6_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "rwkv6.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_rwkv6", cpp_src, cuda_src, ["rwkv6_time_mix_fwd"])
    return _MODULE

def rwkv6_ref(r: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    B, H, N, D = r.shape
    out = torch.empty_like(r)
    s = torch.zeros(B, H, D, device=r.device, dtype=r.dtype)
    r_gate = torch.sigmoid(r)
    k_exp = torch.exp(k)
    decay = torch.exp(-torch.exp(w))
    u_exp = u.view(1, H, 1, D)
    for t in range(N):
        kv = k_exp[:, :, t, :] * v[:, :, t, :]
        y = r_gate[:, :, t, :] * (u_exp.squeeze(2) * kv + s)
        s = decay[:, :, t, :] * s + kv
        out[:, :, t, :] = y
    return out

def _rwkv6_dispatch(r: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    def cuda_fn(r, k, v, w, u):
        mod = _get_module()
        return mod.rwkv6_time_mix_fwd(r.contiguous(), k.contiguous(), v.contiguous(), w.contiguous(), u.contiguous())
    return dispatch(cuda_fn, rwkv6_ref, rwkv6_ref, r, k=k, v=v, w=w, u=u, kernel_name="rwkv6")


class _Rwkv6Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, r, k, v, w, u):
        out = _rwkv6_dispatch(r, k, v, w, u)
        ctx.save_for_backward(*[t for t in (r, k, v, w, u) if isinstance(t, torch.Tensor)])
        ctx.tensors_mask = [isinstance(t, torch.Tensor) for t in (r, k, v, w, u)]
        ctx.non_tensors = [t if not isinstance(t, torch.Tensor) else None for t in (r, k, v, w, u)]
        return out

    @staticmethod
    def backward(ctx: Any, dout):
        saved = list(ctx.saved_tensors)
        inputs = []
        for is_t, nt in zip(ctx.tensors_mask, ctx.non_tensors):
            if is_t:
                t = saved.pop(0)
                inputs.append(t.detach().requires_grad_(t.requires_grad))
            else:
                inputs.append(nt)
        with torch.enable_grad():
            out_p = rwkv6_ref(*inputs)
        tensor_inputs = [inp for inp, is_t in zip(inputs, ctx.tensors_mask) if is_t]
        grads = torch.autograd.grad(out_p, tensor_inputs, dout, allow_unused=True)
        grads = list(grads)
        res = []
        for is_t in ctx.tensors_mask:
            if is_t:
                res.append(grads.pop(0))
            else:
                res.append(None)
        return tuple(res)

def rwkv6(r, k, v, w, u):
    if any(isinstance(t, torch.Tensor) and t.requires_grad for t in (r, k, v, w, u)):
        return _Rwkv6Function.apply(r, k, v, w, u)
    return _rwkv6_dispatch(r, k, v, w, u)
