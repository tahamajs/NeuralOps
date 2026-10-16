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
        with open(curr_dir / "routing_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "routing_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_routing_attention", cpp_src, cuda_src, ["routing_attn_fwd"])
    return _MODULE

def routing_attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, route_weights: torch.Tensor, top_k: int = 32) -> torch.Tensor:
    B, H, N, D = q.shape
    scale = 1.0 / (D ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    mask = route_weights.view(B, H, 1, N) > 0.0
    scores = scores.masked_fill(~mask, float('-inf'))
    probs = F.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.matmul(probs, v)

def routing_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, route_weights: torch.Tensor, top_k: int = 32) -> torch.Tensor:
    def cuda_fn(q, k, v, route_weights, top_k=32):
        mod = _get_module()
        return mod.routing_attn_fwd(q.contiguous(), k.contiguous(), v.contiguous(), route_weights.contiguous(), int(top_k))
    return dispatch(cuda_fn, routing_attention_ref, routing_attention_ref, q, k=k, v=v, route_weights=route_weights, top_k=top_k, kernel_name="routing_attention")
