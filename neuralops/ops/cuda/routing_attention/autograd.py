#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Function for Routing Attention CUDA Kernel.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.autograd import Function

_OPS = None


def _get_routing_ops():
    global _OPS
    if _OPS is not None:
        return _OPS

    try:
        import routing_ops

        _OPS = routing_ops
        return _OPS
    except ImportError:
        pass

    try:
        from torch.utils.cpp_extension import load

        pkg_dir = Path(__file__).parent
        _OPS = load(
            name="routing_ops",
            sources=[str(pkg_dir / "routing_attention.cu"), str(pkg_dir / "binding.cpp")],
            extra_cuda_cflags=["-O3"],  # inherits torch nvcc arches including sm_80, sm_86, sm_89, sm_90
            verbose=False,
        )
        return _OPS
    except Exception:
        return None


class RoutingAttentionFunction(Function):
    """Autograd Function wrapping CUDA routed_attention_forward kernel."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cluster_ids: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(q, k, v, cluster_ids)

        ops = _get_routing_ops()
        if ops is not None and hasattr(ops, "routed_attention_forward"):
            return ops.routed_attention_forward(
                q.contiguous().float(),
                k.contiguous().float(),
                v.contiguous().float(),
                cluster_ids.contiguous().to(torch.int32),
                float(scale),
            )

        # Level 3 PyTorch fallback
        from neuralops.ops.triton.routing_attention_triton import _routing_attention_pytorch

        return _routing_attention_pytorch(q, k, v)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE / Autograd placeholder
        return grad_output, None, None, None, None
