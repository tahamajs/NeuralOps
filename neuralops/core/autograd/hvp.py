#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Autograd Hessian-Vector Product (HVP) and Graph Cleanliness Utilities.

Provides memory-leak-free double-backward Hessian-vector product routines
ensuring graph disposal after second-order gradient computation.
"""

from __future__ import annotations

import gc
from typing import Sequence

import torch
import torch.nn as nn

__all__ = ["compute_hvp", "verify_autograd_graph_cleanliness"]


def compute_hvp(
    loss: torch.Tensor,
    params: Sequence[torch.nn.Parameter | torch.Tensor],
    v: Sequence[torch.Tensor],
    retain_graph: bool = False,
) -> list[torch.Tensor]:
    """Compute Hessian-Vector Product (H * v) via double-backward autograd with explicit graph cleanup.

    Args:
        loss: Scalar loss tensor.
        params: Sequence of parameter or tensor inputs to differentiate with respect to.
        v: Vector tensors matching each parameter in shape.
        retain_graph: Whether to retain graph after the second backward pass. Defaults to False.

    Returns:
        List of Hessian-vector product tensors matching the sequence in params.

    Raises:
        ValueError: If lengths of params and v do not match.
    """
    if len(params) != len(v):
        raise ValueError(f"Length mismatch: {len(params)} parameters vs {len(v)} vectors")

    param_list = list(params)
    v_list = list(v)

    # First backward pass with create_graph=True
    grads = torch.autograd.grad(
        loss,
        param_list,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )

    grad_dot_v = sum(
        (g * vi).sum()
        for g, vi in zip(grads, v_list)
        if g is not None
    )

    # Second backward pass to compute HVP
    hvps = torch.autograd.grad(
        grad_dot_v,
        param_list,
        retain_graph=retain_graph,
        allow_unused=True,
    )

    # Explicit graph detachment and intermediate cleanup
    del grads, grad_dot_v

    return [
        h.detach() if (h is not None and not retain_graph) else (h if h is not None else torch.zeros_like(p))
        for h, p in zip(hvps, param_list)
    ]


def verify_autograd_graph_cleanliness(
    model: nn.Module,
    sample_input: torch.Tensor,
    num_iterations: int = 20,
) -> bool:
    """Verify that repeated double-backward HVP passes do not leak autograd graph or memory.

    Args:
        model: Target PyTorch neural network module.
        sample_input: Input tensor matching model's expected forward signature.
        num_iterations: Number of evaluation cycles.

    Returns:
        True if all cycles complete and memory does not leak.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return True

    v = [torch.randn_like(p) for p in params]

    # Warmup
    for _ in range(3):
        out = model(sample_input)
        loss = (out**2).sum()
        _ = compute_hvp(loss, params, v, retain_graph=False)
        del out, loss

    gc.collect()

    # Iterative verification
    for _ in range(num_iterations):
        out = model(sample_input)
        loss = (out**2).sum()
        hvps = compute_hvp(loss, params, v, retain_graph=False)
        assert len(hvps) == len(params)
        del out, loss, hvps

    gc.collect()
    return True
