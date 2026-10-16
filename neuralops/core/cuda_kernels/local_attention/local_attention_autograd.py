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
        with open(curr_dir / "local_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "local_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_local_attention", cpp_src, cuda_src, ["local_attn_fwd"])
    return _MODULE

def local_attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int = 64) -> torch.Tensor:
    B, H, N, D = q.shape
    scale = 1.0 / (D ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    idx = torch.arange(N, device=q.device)
    dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
    mask = dist <= window_size
    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v)

def local_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int = 64) -> torch.Tensor:
    def cuda_fn(q, k, v, window_size=64):
        mod = _get_module()
        return mod.local_attn_fwd(q.contiguous(), k.contiguous(), v.contiguous(), int(window_size))
    return dispatch(cuda_fn, local_attention_ref, local_attention_ref, q, k, v, window_size=window_size, kernel_name="local_attention")
