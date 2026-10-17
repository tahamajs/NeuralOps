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
        with open(curr_dir / "cluster_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "cluster_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_cluster_attention", cpp_src, cuda_src, ["cluster_attn_fwd"])
    return _MODULE

def cluster_attention_ref(q: torch.Tensor, cluster_centroids: torch.Tensor, v: torch.Tensor, cluster_indices: torch.Tensor) -> torch.Tensor:
    B, H, N, D = q.shape
    scale = 1.0 / (D ** 0.5)
    scores = torch.matmul(q, cluster_centroids.transpose(-2, -1)) * scale
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v)

def cluster_attention(q: torch.Tensor, cluster_centroids: torch.Tensor, v: torch.Tensor, cluster_indices: torch.Tensor) -> torch.Tensor:
    def cuda_fn(q, cluster_centroids, v, cluster_indices):
        mod = _get_module()
        return mod.cluster_attn_fwd(q.contiguous(), cluster_centroids.contiguous(), v.contiguous(), cluster_indices.contiguous().int())
    if q.shape[-1] > 128:
        cuda_fn = None
    return dispatch(cuda_fn, cluster_attention_ref, cluster_attention_ref, q, cluster_centroids=cluster_centroids, v=v, cluster_indices=cluster_indices, kernel_name="cluster_attention")
