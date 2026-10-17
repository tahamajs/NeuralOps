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
        with open(curr_dir / "decay_linear_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "decay_linear_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_decay_linear_attention", cpp_src, cuda_src, ["decay_linear_fwd"])
    return _MODULE

def decay_linear_attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    B, H, N, D = q.shape
    out = torch.empty_like(q)
    s = torch.zeros(B, H, D, device=q.device, dtype=q.dtype)
    g = gamma.view(1, H, 1, D)
    for t in range(N):
        s = g.squeeze(2) * s + k[:, :, t, :] * v[:, :, t, :]
        out[:, :, t, :] = q[:, :, t, :] * s
    return out

def _decay_linear_attention_dispatch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    def cuda_fn(q, k, v, gamma):
        mod = _get_module()
        return mod.decay_linear_fwd(q.contiguous(), k.contiguous(), v.contiguous(), gamma.contiguous())
    return dispatch(cuda_fn, decay_linear_attention_ref, decay_linear_attention_ref, q, k=k, v=v, gamma=gamma, kernel_name="decay_linear_attention")


class _Decay_linear_attentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q, k, v, gamma):
        out = _decay_linear_attention_dispatch(q, k, v, gamma)
        ctx.save_for_backward(*[t for t in (q, k, v, gamma) if isinstance(t, torch.Tensor)])
        ctx.tensors_mask = [isinstance(t, torch.Tensor) for t in (q, k, v, gamma)]
        ctx.non_tensors = [t if not isinstance(t, torch.Tensor) else None for t in (q, k, v, gamma)]
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
            out_p = decay_linear_attention_ref(*inputs)
        tensor_inputs = [inp for inp, is_t in zip(inputs, ctx.tensors_mask) if is_t]
        dout_val = douts[0] if len(douts) == 1 else douts
        grads = torch.autograd.grad(out_p, tensor_inputs, dout_val, allow_unused=True)
        grads = list(grads)
        res = []
        for is_t in ctx.tensors_mask:
            if is_t:
                res.append(grads.pop(0))
            else:
                res.append(None)
        return tuple(res)

def decay_linear_attention(q, k, v, gamma):
    if any(isinstance(t, torch.Tensor) and t.requires_grad for t in (q, k, v, gamma)):
        return _Decay_linear_attentionFunction.apply(q, k, v, gamma)
    return _decay_linear_attention_dispatch(q, k, v, gamma)
