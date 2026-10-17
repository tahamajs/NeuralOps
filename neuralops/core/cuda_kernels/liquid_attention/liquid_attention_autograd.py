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
        with open(curr_dir / "liquid_attention_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "liquid_attention.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_liquid_attention", cpp_src, cuda_src, ["liquid_attention_fwd"])
    return _MODULE

def liquid_attention_ref(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    prev_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, L, D = x.shape
    dt = torch.clamp(delta.float(), min=0.0)
    a_val = torch.abs(A.float())
    decay = torch.exp(-torch.clamp(dt * a_val, min=0.0))

    h = prev_state.float().clone() if prev_state is not None else torch.zeros(B, D, device=x.device, dtype=torch.float32)
    ys = []
    for t in range(L):
        in_t = x[:, t, :].float()
        dec_t = decay[:, t, :]
        h = dec_t * h + (1.0 - dec_t) * in_t
        h = torch.clamp(h, -65504.0, 65504.0)
        sig = torch.sigmoid(in_t)
        ys.append((h * sig).to(x.dtype))

    out = torch.stack(ys, dim=1)
    return out, h.to(x.dtype)

def _liquid_attention_dispatch(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    prev_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    def cuda_fn(x, delta, A, prev_state=None, **kwargs):
        mod = _get_module()
        res = mod.liquid_attention_fwd(
            x.contiguous(),
            delta.contiguous(),
            A.contiguous(),
            prev_state.contiguous() if prev_state is not None else None,
        )
        return res[0], res[1]

    return dispatch(
        cuda_fn,
        liquid_attention_ref,
        liquid_attention_ref,
        x,
        delta=delta,
        A=A,
        prev_state=prev_state,
        kernel_name="liquid_attention",
    )


class _Liquid_attentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x, delta, A):
        out = _liquid_attention_dispatch(x, delta, A)
        ctx.save_for_backward(*[t for t in (x, delta, A) if isinstance(t, torch.Tensor)])
        ctx.tensors_mask = [isinstance(t, torch.Tensor) for t in (x, delta, A)]
        ctx.non_tensors = [t if not isinstance(t, torch.Tensor) else None for t in (x, delta, A)]
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
            out_p = liquid_attention_ref(*inputs)
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

def liquid_attention(x, delta, A):
    if any(isinstance(t, torch.Tensor) and t.requires_grad for t in (x, delta, A)):
        return _Liquid_attentionFunction.apply(x, delta, A)
    return _liquid_attention_dispatch(x, delta, A)
