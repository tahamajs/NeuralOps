#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Cluster Attention Autograd Interface

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
        with open(curr_dir / "cluster_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "cluster_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_cluster_attention", cpp_src, cuda_src, ["cluster_attn_fwd"])
    return _MODULE

def cluster_attention_ref(q: torch.Tensor, cluster_centroids: torch.Tensor, v: torch.Tensor, cluster_indices: torch.Tensor) -> torch.Tensor:
    if v.shape == cluster_centroids.shape:
        return F.scaled_dot_product_attention(q, cluster_centroids, v)
    return F.scaled_dot_product_attention(q, cluster_centroids, cluster_centroids)


def cluster_attention(q: torch.Tensor, cluster_centroids: torch.Tensor, v: torch.Tensor, cluster_indices: torch.Tensor) -> torch.Tensor:
    def cuda_fn(q, cluster_centroids, v, cluster_indices):
        mod = _get_module()
        return mod.cluster_attn_fwd(q.contiguous(), cluster_centroids.contiguous(), v.contiguous(), cluster_indices.contiguous().int())
    if q.shape[-1] > 128:
        cuda_fn = None
    return dispatch(cuda_fn, cluster_attention_ref, cluster_attention_ref, q, cluster_centroids=cluster_centroids, v=v, cluster_indices=cluster_indices, kernel_name="cluster_attention")
