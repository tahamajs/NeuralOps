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

def liquid_attention(
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
