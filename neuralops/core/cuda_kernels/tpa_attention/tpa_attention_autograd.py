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
        with open(curr_dir / "tpa_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "tpa.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_tpa", cpp_src, cuda_src, ["tpa_fwd"])
    return _MODULE

def tpa_attention_ref(h_states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # weights: [B, N], h_states: [B, N, D] -> [B, D]
    return torch.bmm(weights.unsqueeze(1), h_states).squeeze(1)

def tpa_attention(h_states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    def cuda_fn(h_states, weights):
        mod = _get_module()
        return mod.tpa_fwd(h_states.contiguous(), weights.contiguous())
    return dispatch(cuda_fn, tpa_attention_ref, tpa_attention_ref, h_states, weights=weights, kernel_name="tpa_attention")
