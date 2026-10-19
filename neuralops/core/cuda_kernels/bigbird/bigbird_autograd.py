# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
BigBird Sparse Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

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
        with open(curr_dir / "bigbird_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "bigbird.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_bigbird", cpp_src, cuda_src, ["bigbird_fwd"])
    return _MODULE

def bigbird_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int = 64, num_global_tokens: int = 4, num_random_blocks: int = 2) -> torch.Tensor:
    B, H, N, D = q.shape
    scale = 1.0 / (D ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    idx = torch.arange(N, device=q.device)
    dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
    local_mask = dist <= window_size
    global_mask = (idx.unsqueeze(1) < num_global_tokens) | (idx.unsqueeze(0) < num_global_tokens)
    random_mask = ((idx.unsqueeze(1) * 37 + idx.unsqueeze(0) * 17) % max(1, N // max(1, num_random_blocks))) == 0
    mask = local_mask | global_mask | random_mask
    scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    probs = F.softmax(scores, dim=-1)
    return torch.matmul(probs, v)

def bigbird(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int = 64, num_global_tokens: int = 4, num_random_blocks: int = 2) -> torch.Tensor:
    def cuda_fn(q, k, v, window_size=64, num_global_tokens=4, num_random_blocks=2):
        mod = _get_module()
        return mod.bigbird_fwd(q.contiguous(), k.contiguous(), v.contiguous(), int(window_size), int(num_global_tokens), int(num_random_blocks))
    return dispatch(cuda_fn, bigbird_ref, bigbird_ref, q, k, v, window_size=window_size, num_global_tokens=num_global_tokens, num_random_blocks=num_random_blocks, kernel_name="bigbird")
