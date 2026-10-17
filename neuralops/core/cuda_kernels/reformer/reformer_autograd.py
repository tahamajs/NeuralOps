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
        with open(curr_dir / "reformer_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "reformer.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_reformer", cpp_src, cuda_src, ["reformer_fwd"])
    return _MODULE

def reformer_ref(qk: torch.Tensor, v: torch.Tensor, hash_buckets: torch.Tensor) -> torch.Tensor:
    B, H, N, D = qk.shape
    scale = 1.0 / (D ** 0.5)
    scores = torch.matmul(qk, qk.transpose(-2, -1)) * scale
    same_bucket = (hash_buckets.unsqueeze(2) == hash_buckets.unsqueeze(1))
    scores = scores.masked_fill(~same_bucket.unsqueeze(1), float('-inf'))
    probs = F.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.matmul(probs, v)

def reformer(qk: torch.Tensor, v: torch.Tensor, hash_buckets: torch.Tensor) -> torch.Tensor:
    cuda_fn = None
    if qk.shape[-1] <= 128:
        def _cuda_fn(qk, v, hash_buckets):
            mod = _get_module()
            return mod.reformer_fwd(qk.contiguous(), v.contiguous(), hash_buckets.contiguous().int())
        cuda_fn = _cuda_fn
    return dispatch(cuda_fn, reformer_ref, reformer_ref, qk, v=v, hash_buckets=hash_buckets, kernel_name="reformer")
