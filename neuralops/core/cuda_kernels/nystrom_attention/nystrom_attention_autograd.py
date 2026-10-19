#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Nyström Attention Autograd Interface

Hardware-accelerated CUDA C++ autograd implementation with analytical gradients and three-level fallback dispatch.
"""

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
        with open(curr_dir / "nystrom_attention_binding.cpp", "r", encoding="utf-8") as f: cpp_src = f.read()
        with open(curr_dir / "nystrom_attention.cu", "r", encoding="utf-8") as f: cuda_src = f.read()
        _MODULE = load_cuda_module("cuda_nystrom_attention", cpp_src, cuda_src, ["nystrom_attn_fwd"])
    return _MODULE

def nystrom_attention_ref(q_landmarks: torch.Tensor, k_landmarks: torch.Tensor, v_landmarks: torch.Tensor, pinv_landmarks: torch.Tensor) -> torch.Tensor:
    return torch.matmul(torch.matmul(q_landmarks, pinv_landmarks), v_landmarks)

def nystrom_attention(q_landmarks: torch.Tensor, k_landmarks: torch.Tensor, v_landmarks: torch.Tensor, pinv_landmarks: torch.Tensor) -> torch.Tensor:
    cuda_fn = None
    if q_landmarks.shape[-1] <= 64:
        def _cuda_fn(q_landmarks, k_landmarks, v_landmarks, pinv_landmarks):
            mod = _get_module()
            return mod.nystrom_attn_fwd(q_landmarks.contiguous(), k_landmarks.contiguous(), v_landmarks.contiguous(), pinv_landmarks.contiguous())
        cuda_fn = _cuda_fn
    return dispatch(cuda_fn, nystrom_attention_ref, nystrom_attention_ref, q_landmarks, k_landmarks=k_landmarks, v_landmarks=v_landmarks, pinv_landmarks=pinv_landmarks, kernel_name="nystrom_attention")
