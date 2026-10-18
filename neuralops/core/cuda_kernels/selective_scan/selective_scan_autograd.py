"""
Selective Scan (Mamba-1 S6) Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.

Author: Taha Majlesi
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None


def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "selective_scan_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "selective_scan.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()

        _MODULE = load_cuda_module(
            name="cuda_selective_scan",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["selective_scan_fwd", "selective_scan_bwd"],
        )
    return _MODULE


def selective_scan_pytorch_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference implementation with high-precision float32 accumulation."""
    b, d, l = u.shape
    d_state = A.shape[1]

    u_f = u.float()
    delta_f = delta.float()
    A_f = A.float()
    B_f = B.float()
    C_f = C.float()
    D_f = D.float() if D is not None and D.numel() > 0 else None

    h = torch.zeros(b, d, d_state, device=u.device, dtype=torch.float32)
    ys = []

    for i in range(l):
        u_t = u_f[:, :, i]  # [b, d]
        delta_t = delta_f[:, :, i]  # [b, d]

        # dA = exp(delta_t * A) -> [b, d, d_state]
        dA = torch.exp(torch.clamp(delta_t.unsqueeze(-1) * A_f.unsqueeze(0), -20.0, 20.0))
        # dB = delta_t * B[:, :, i] -> [b, d, d_state]
        dB = delta_t.unsqueeze(-1) * B_f[:, :, i].unsqueeze(1)

        h = dA * h + dB * u_t.unsqueeze(-1)
        y = torch.einsum("bdn,bn->bd", h, C_f[:, :, i])

        if D_f is not None:
            y = y + u_t * D_f.unsqueeze(0)
        ys.append(y)

    out = torch.stack(ys, dim=-1).to(u.dtype)
    return out, h.to(u.dtype)


class _SelectiveScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None):
        mod = _get_module()
        if mod is not None and u.is_cuda and A.shape[1] <= 128:
            d_tensor = D if D is not None and D.numel() > 0 else torch.empty(0, device=u.device, dtype=u.dtype)
            out, last_state, states = mod.selective_scan_fwd(u, delta, A, B, C, d_tensor, True)
            ctx.save_for_backward(u, delta, A, B, C, d_tensor, states)
            ctx.has_d = D is not None and D.numel() > 0
            ctx.is_py = False
            return out, last_state

        # PyTorch reference forward under autograd
        u_p = u.detach().requires_grad_(u.requires_grad)
        delta_p = delta.detach().requires_grad_(delta.requires_grad)
        A_p = A.detach().requires_grad_(A.requires_grad)
        B_p = B.detach().requires_grad_(B.requires_grad)
        C_p = C.detach().requires_grad_(C.requires_grad)
        D_p = D.detach().requires_grad_(D.requires_grad) if D is not None and D.numel() > 0 else None

        with torch.enable_grad():
            out, last_state = selective_scan_pytorch_ref(u_p, delta_p, A_p, B_p, C_p, D_p)

        ctx.py_tensors = (u_p, delta_p, A_p, B_p, C_p, D_p, out, last_state)
        ctx.is_py = True
        return out.detach(), last_state.detach()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        dout = grad_outputs[0] if len(grad_outputs) > 0 else None
        dlast_state = grad_outputs[1] if len(grad_outputs) > 1 else None
        if getattr(ctx, "is_py", False):
            u_p, delta_p, A_p, B_p, C_p, D_p, out, last_state = ctx.py_tensors
            inputs = [u_p, delta_p, A_p, B_p, C_p]
            if D_p is not None:
                inputs.append(D_p)
            outputs = [out]
            grad_outputs = [dout]
            if last_state is not None and dlast_state is not None:
                outputs.append(last_state)
                grad_outputs.append(dlast_state)
            grads = torch.autograd.grad(outputs, inputs, grad_outputs, allow_unused=True)
            du, ddelta, dA, dB, dC = grads[:5]
            dD = grads[5] if len(grads) > 5 else None
            return du, ddelta, dA, dB, dC, dD

        u, delta, A, B, C, D, states = ctx.saved_tensors
        mod = _get_module()
        du, ddelta, dA, dB, dC, dD = mod.selective_scan_bwd(u, delta, A, B, C, D, states, dout)
        return du, ddelta, dA, dB, dC, dD if ctx.has_d else None


def _selective_scan_forward_dispatch(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mod = _get_module()

    def cuda_fn(u, delta, A, B, C, D):
        d_tensor = D if D is not None and D.numel() > 0 else torch.empty(0, device=u.device, dtype=u.dtype)
        out, last_state, _ = mod.selective_scan_fwd(u, delta, A, B, C, d_tensor, False)
        return out, last_state

    triton_fn = None
    try:
        from neuralops.core.triton_kernels.selective_scan import selective_scan_fn

        def _triton_wrapper(u, delta, A, B, C, D):
            out = selective_scan_fn(u, delta, A, B, C, D, delta_softplus=False)
            last_state = torch.zeros(u.size(0), u.size(1), A.size(1), device=u.device, dtype=u.dtype)
            return out, last_state

        triton_fn = _triton_wrapper
    except Exception:
        pass

    return dispatch(
        cuda_fn if (mod is not None and A.shape[1] <= 128) else None,
        triton_fn,
        selective_scan_pytorch_ref,
        u,
        delta,
        A,
        B,
        C,
        D,
        kernel_name="selective_scan",
    )


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Selective Scan with autograd backward pass and 3-level fallback."""
    if any(t.requires_grad for t in (u, delta, A, B, C, D) if isinstance(t, torch.Tensor)):
        return _SelectiveScanFunction.apply(u, delta, A, B, C, D)

    return _selective_scan_forward_dispatch(u, delta, A, B, C, D)
