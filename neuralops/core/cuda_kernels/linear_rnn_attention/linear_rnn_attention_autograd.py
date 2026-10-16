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
        with open(curr_dir / "linear_rnn_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "linear_rnn_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_linear_rnn_attention", cpp_src, cuda_src, ["linear_rnn_fwd"])
    return _MODULE

def linear_rnn_attention_ref(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    B, N, D = x.shape
    out = torch.empty_like(x)
    h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
    for t in range(N):
        h = a[:, t, :] * h + b[:, t, :] * x[:, t, :]
        out[:, t, :] = h
    return out

def linear_rnn_attention(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    def cuda_fn(x, a, b):
        mod = _get_module()
        return mod.linear_rnn_fwd(x.contiguous(), a.contiguous(), b.contiguous())
    return dispatch(cuda_fn, linear_rnn_attention_ref, linear_rnn_attention_ref, x, a=a, b=b, kernel_name="linear_rnn_attention")
