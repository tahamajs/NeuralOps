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
        with open(curr_dir / "mamba_block_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "mamba_block.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_mamba_block", cpp_src, cuda_src, ["mamba_step_fwd"])
    return _MODULE

def mamba_block_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    prev_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dt = torch.exp(delta) # [B, D]
    dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0)) # [B, D, S]
    dB = dt.unsqueeze(-1) * B_mat.unsqueeze(1) # [B, D, S]
    next_state = dA * prev_state + dB * u.unsqueeze(-1)
    out = (next_state * C_mat.unsqueeze(1)).sum(dim=-1)
    return out, next_state

def mamba_block(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B_mat: torch.Tensor,
    C_mat: torch.Tensor,
    prev_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    def cuda_fn(u, delta, A, B_mat, C_mat, prev_state, **kwargs):
        mod = _get_module()
        res = mod.mamba_step_fwd(
            u.contiguous(),
            delta.contiguous(),
            A.contiguous(),
            B_mat.contiguous(),
            C_mat.contiguous(),
            prev_state.contiguous(),
        )
        return res[0], res[1]

    return dispatch(
        cuda_fn,
        mamba_block_ref,
        mamba_block_ref,
        u,
        delta,
        A,
        B_mat,
        C_mat,
        prev_state,
        kernel_name="mamba_block",
    )
