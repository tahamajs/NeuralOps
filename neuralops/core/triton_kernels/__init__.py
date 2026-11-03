#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Backward compatibility shim for neuralops.ops.triton."""

from __future__ import annotations

from typing import Any
import sys

HAS_TRITON = True

from neuralops.ops.triton._common import triton_dispatch
from neuralops.ops.triton.cosformer_triton_kernel import triton_cosformer_linear_attention, triton_cosformer_linear_attention_pytorch_ref
from neuralops.ops.triton.block_sparse import fused_dsa_sparse_attn_triton
from neuralops.ops.triton.linear_attn import fwd as fast_linear_attention

__all__ = [
    "HAS_TRITON",
    "triton_dispatch",
    "triton_cosformer_linear_attention",
    "triton_cosformer_linear_attention_pytorch_ref",
    "fused_dsa_sparse_attn_triton",
    "fast_linear_attention",
]

def __getattr__(name: str) -> Any:
    if name == "HAS_TRITON":
        return True
    if name == "triton_dispatch":
        return triton_dispatch
    if name == "triton_cosformer_linear_attention":
        return triton_cosformer_linear_attention
    if name == "fused_dsa_sparse_attn_triton":
        return fused_dsa_sparse_attn_triton
    if name == "fast_linear_attention":
        return fast_linear_attention
    import neuralops.ops.triton as _ops_triton
    return getattr(_ops_triton, name)

def __dir__() -> list[str]:
    import neuralops.ops.triton as _ops_triton
    return list(_ops_triton.__dir__()) + ["HAS_TRITON", "fast_linear_attention", "triton_dispatch", "triton_cosformer_linear_attention", "fused_dsa_sparse_attn_triton"]
