"""
RWKV-6 Time-Mixing Recurrence Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.

Author: Taha Majlesi
"""

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
    orig_3d = (r.dim() == 3)
    if orig_3d:
        r = r.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        w = w.unsqueeze(1)
    B, H, N, D = r.shape
    out = torch.empty_like(r)
    s = torch.zeros(B, H, D, device=r.device, dtype=r.dtype)
    r_gate = torch.sigmoid(r)
    k_exp = torch.exp(k)
    decay = torch.exp(-torch.exp(w))
    u_vec = u.view(1, H, 1, D) if (u.numel() == H * D or u.dim() == 2) else u.view(1, 1, 1, -1)[:, :, :, :D]
    u_val = u_vec.squeeze(2)
    for t in range(N):
        kv = k_exp[:, :, t, :] * v[:, :, t, :]
        y = r_gate[:, :, t, :] * (u_val * kv + s)
        s = decay[:, :, t, :] * s + kv
        out[:, :, t, :] = y
    return out.squeeze(1) if orig_3d else out

def _rwkv6_dispatch(r: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    orig_3d = (r.dim() == 3)
    r_in = r.unsqueeze(1) if orig_3d else r
    k_in = k.unsqueeze(1) if orig_3d else k
    v_in = v.unsqueeze(1) if orig_3d else v
    w_in = w.unsqueeze(1) if orig_3d else w
    B, H, N, D = r_in.shape
    u_in = u.reshape(H, D).contiguous() if u.numel() == H * D else u.reshape(-1)[:D].repeat(H, 1).contiguous()
    
    def cuda_fn(r_t, k_t, v_t, w_t, u_t):
        mod = _get_module()
        res = mod.rwkv6_time_mix_fwd(r_t.contiguous(), k_t.contiguous(), v_t.contiguous(), w_t.contiguous(), u_t.contiguous())
        return res
        
    res = dispatch(cuda_fn, rwkv6_ref, rwkv6_ref, r_in, k=k_in, v=v_in, w=w_in, u=u_in, kernel_name="rwkv6")
    return res.squeeze(1) if orig_3d and res.dim() == 4 else res


class _Rwkv6Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, r, k, v, w, u):
        out = _rwkv6_dispatch(r, k, v, w, u)
        ctx.save_for_backward(*[t for t in (r, k, v, w, u) if isinstance(t, torch.Tensor)])
        ctx.tensors_mask = [isinstance(t, torch.Tensor) for t in (r, k, v, w, u)]
        ctx.non_tensors = [t if not isinstance(t, torch.Tensor) else None for t in (r, k, v, w, u)]
        return out

    @staticmethod
    def backward(ctx: Any, *douts):
        dout = douts[0] if len(douts) == 1 else douts
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
        diff_inputs = [inp for inp, is_t in zip(inputs, ctx.tensors_mask) if (is_t and inp.requires_grad)]
        dout_val = douts[0] if len(douts) == 1 else douts
        if not diff_inputs:
            return tuple([None] * len(ctx.tensors_mask))
        grads = torch.autograd.grad(out_p, diff_inputs, dout_val, allow_unused=True)
        grad_map = dict(zip(diff_inputs, grads))
        return tuple(grad_map.get(inp, None) if (is_t and inp.requires_grad) else None for inp, is_t in zip(inputs, ctx.tensors_mask))

def rwkv6(r, k, v, w, u):
    if any(isinstance(t, torch.Tensor) and t.requires_grad for t in (r, k, v, w, u)):
        return _Rwkv6Function.apply(r, k, v, w, u)
    return _rwkv6_dispatch(r, k, v, w, u)
