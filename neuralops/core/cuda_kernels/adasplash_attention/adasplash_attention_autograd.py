from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F
from neuralops.core.cuda_kernels._bindings import load_cuda_module
from neuralops.core.cuda_kernels._fallback import dispatch

_MODULE = None
def _get_module():
    global _MODULE
    if _MODULE is None:
        curr_dir = Path(__file__).parent
        with open(curr_dir / "adasplash_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "adasplash.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_adasplash", cpp_src, cuda_src, ["adasplash_fwd"])
    return _MODULE

def adasplash_attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, threshold: float = 0.1, window_size: int = 64) -> torch.Tensor:
    B, H, N, D = q.shape
    scale = 1.0 / (D ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    causal_mask = torch.triu(torch.full((N, N), float('-inf'), device=q.device), diagonal=1)
    window_mask = torch.tril(torch.full((N, N), float('-inf'), device=q.device), diagonal=-window_size)
    scores = scores + causal_mask + window_mask
    mask = (scores >= threshold) | (torch.eye(N, device=q.device).bool())
    scores = scores.masked_fill(~mask, float('-inf'))
    probs = F.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.matmul(probs, v)

def adasplash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, threshold: float = 0.1, window_size: int = 64) -> torch.Tensor:
    def cuda_fn(q, k, v, threshold=0.1, window_size=64):
        mod = _get_module()
        return mod.adasplash_fwd(q.contiguous(), k.contiguous(), v.contiguous(), float(threshold), int(window_size))
    return dispatch(cuda_fn, adasplash_attention_ref, adasplash_attention_ref, q, k, v, threshold=threshold, window_size=window_size, kernel_name="adasplash_attention")
