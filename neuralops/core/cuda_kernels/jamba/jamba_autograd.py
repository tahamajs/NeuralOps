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
        with open(curr_dir / "jamba_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "jamba.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_jamba", cpp_src, cuda_src, ["jamba_mix_fwd"])
    return _MODULE

def jamba_ref(mamba_out: torch.Tensor, attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    g = torch.sigmoid(gate)
    return g * mamba_out + (1.0 - g) * attn_out

def jamba(mamba_out: torch.Tensor, attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    def cuda_fn(mamba_out, attn_out, gate):
        mod = _get_module()
        return mod.jamba_mix_fwd(mamba_out.contiguous(), attn_out.contiguous(), gate.contiguous())
    return dispatch(cuda_fn, jamba_ref, jamba_ref, mamba_out, attn_out=attn_out, gate=gate, kernel_name="jamba")
