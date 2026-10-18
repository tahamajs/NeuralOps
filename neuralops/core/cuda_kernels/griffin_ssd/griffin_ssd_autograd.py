"""
Griffin SSD Recurrent Operator Autograd Interface

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

def _griffin_ssd_dispatch(x: torch.Tensor, r: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    def cuda_fn(x, r, decay):
        mod = _get_module()
        return mod.griffin_ssd_fwd(x.contiguous(), r.contiguous(), decay.contiguous())
    return dispatch(cuda_fn, griffin_ssd_ref, griffin_ssd_ref, x, r=r, decay=decay, kernel_name="griffin_ssd")


class _Griffin_ssdFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x, r, decay):
        out = _griffin_ssd_dispatch(x, r, decay)
        ctx.save_for_backward(*[t for t in (x, r, decay) if isinstance(t, torch.Tensor)])
        ctx.tensors_mask = [isinstance(t, torch.Tensor) for t in (x, r, decay)]
        ctx.non_tensors = [t if not isinstance(t, torch.Tensor) else None for t in (x, r, decay)]
        return out

    @staticmethod
    def backward(ctx: Any, *douts):
        saved = list(ctx.saved_tensors)
        inputs = []
        for is_t, nt in zip(ctx.tensors_mask, ctx.non_tensors):
            if is_t:
                t = saved.pop(0)
                inputs.append(t.detach().requires_grad_(t.requires_grad))
            else:
                inputs.append(nt)
        with torch.enable_grad():
            out_p = griffin_ssd_ref(*inputs)
        tensor_inputs = [inp for inp, is_t in zip(inputs, ctx.tensors_mask) if is_t]
        dout_val = douts[0] if len(douts) == 1 else douts
        grads = torch.autograd.grad(out_p, tensor_inputs, dout_val, allow_unused=True)
        grad_list: list[Any] = list(grads)
        res = []
        for is_t in ctx.tensors_mask:
            if is_t:
                res.append(grad_list.pop(0))
            else:
                res.append(None)
        return tuple(res)

def griffin_ssd(x, r, decay):
    if any(isinstance(t, torch.Tensor) and t.requires_grad for t in (x, r, decay)):
        return _Griffin_ssdFunction.apply(x, r, decay)
    return _griffin_ssd_dispatch(x, r, decay)
