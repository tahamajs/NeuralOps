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
        with open(curr_dir / "mamba2_block_binding.cpp", "r", encoding="utf-8") as f:
            cpp_src = f.read()
        with open(curr_dir / "mamba2_block.cu", "r", encoding="utf-8") as f:
            cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_mamba2_block", cpp_src, cuda_src, ["mamba2_block_fwd"])
    return _MODULE

def mamba2_block_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    D_val: torch.Tensor | None = None,
    prev_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, D, L = u.shape
    N = B_mat.shape[1]
    dt = torch.clamp(delta.float(), min=0.0) # [B, D, L]
    a_val = torch.abs(A.float()) # [D]
    dA = torch.exp(-torch.clamp(dt * a_val.unsqueeze(0).unsqueeze(-1), min=0.0)) # [B, D, L]

    h = prev_state.float().clone() if prev_state is not None else torch.zeros(B, D, N, device=u.device, dtype=torch.float32)
    ys = []

    for t in range(L):
        u_t = u[:, :, t].float() # [B, D]
        dt_t = dt[:, :, t]       # [B, D]
        dA_t = dA[:, :, t]       # [B, D]
        b_t = B_mat[:, :, t].float() # [B, N]
        c_t = C_mat[:, :, t].float() # [B, N]

        # h: [B, D, N] = dA_t * h + dt_t * u_t (outer) b_t
        h = dA_t.unsqueeze(-1) * h + (dt_t * u_t).unsqueeze(-1) * b_t.unsqueeze(1)
        h = torch.clamp(h, -65504.0, 65504.0)

        # y_t: [B, D] = (h * c_t.unsqueeze(1)).sum(-1)
        y_t = (h * c_t.unsqueeze(1)).sum(dim=-1)
        if D_val is not None:
            y_t = y_t + D_val.float().unsqueeze(0) * u_t
        ys.append(y_t.to(u.dtype))

    out = torch.stack(ys, dim=-1) # [B, D, L]
    return out, h.to(u.dtype)

def mamba2_block(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    D_val: torch.Tensor | None = None,
    prev_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    def cuda_fn(u, delta, A, B_mat, C_mat, D_val=None, prev_state=None, **kwargs):
        mod = _get_module()
        res = mod.mamba2_block_fwd(
            u.contiguous(),
            delta.contiguous(),
            A.contiguous(),
            B_mat.contiguous(),
            C_mat.contiguous(),
            D_val.contiguous() if D_val is not None else None,
            prev_state.contiguous() if prev_state is not None else None,
        )
        return res[0], res[1]

    return dispatch(
        cuda_fn,
        mamba2_block_ref,
        mamba2_block_ref,
        u,
        delta=delta,
        A=A,
        B_mat=B_mat,
        C_mat=C_mat,
        D_val=D_val,
        prev_state=prev_state,
        kernel_name="mamba2_block",
    )
