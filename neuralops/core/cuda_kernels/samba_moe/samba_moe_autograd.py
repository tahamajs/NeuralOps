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
        with open(curr_dir / "samba_moe_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "samba_moe.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_samba_moe", cpp_src, cuda_src, ["samba_moe_combine_fwd"])
    return _MODULE

def samba_moe_ref(expert_out1: torch.Tensor, expert_out2: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
    w1 = routing_weights[..., 0:1]
    w2 = routing_weights[..., 1:2]
    return w1 * expert_out1 + w2 * expert_out2

def samba_moe(expert_out1: torch.Tensor, expert_out2: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
    def cuda_fn(expert_out1, expert_out2, routing_weights):
        mod = _get_module()
        return mod.samba_moe_combine_fwd(expert_out1.contiguous(), expert_out2.contiguous(), routing_weights.contiguous())
    return dispatch(cuda_fn, samba_moe_ref, samba_moe_ref, expert_out1, expert_out2, routing_weights, kernel_name="samba_moe")
