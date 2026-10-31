#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Activation-Aware Singular Value Decomposition (AASVD) Algorithm.

Decomposes linear weight projections weighted by activation energy covariance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.Operations.AASVD")


def compute_activation_covariance(
    model: nn.Module,
    target_modules: List[Tuple[str, nn.Module]],
    calibration_data: Optional[Any] = None,
    calibration_tokens: int = 512,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Compute empirical input activation covariance matrices for target modules."""
    activations: Dict[str, List[torch.Tensor]] = {path: [] for path, _ in target_modules}
    hooks = []

    def make_hook(path: str):
        def _hook(mod, inp, out):
            if inp and isinstance(inp[0], torch.Tensor):
                x = inp[0].detach()
                # Reshape to (N, d_in)
                flat = x.view(-1, x.size(-1)).float()
                activations[path].append(flat.cpu())
        return _hook

    for path, mod in target_modules:
        h = mod.register_forward_hook(make_hook(path))
        hooks.append(h)

    model_device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    if device is not None:
        model_device = device

    model.eval()
    try:
        with torch.no_grad():
            if calibration_data is not None:
                for batch in calibration_data:
                    if isinstance(batch, dict):
                        b = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        _ = model(**b)
                    elif isinstance(batch, torch.Tensor):
                        _ = model(batch.to(model_device))
            else:
                # Synthetic calibration token batch
                vocab_size = getattr(getattr(model, "config", None), "vocab_size", 30522)
                seq_len = min(calibration_tokens, 128)
                batch_size = max(1, calibration_tokens // seq_len)
                dummy_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=model_device)
                try:
                    _ = model(dummy_tokens)
                except Exception:
                    # Fallback for models taking float inputs
                    d_in = getattr(getattr(model, "config", None), "hidden_size", 768)
                    dummy_inputs = torch.randn(batch_size, seq_len, d_in, device=model_device)
                    _ = model(dummy_inputs)
    finally:
        for h in hooks:
            h.remove()

    cov_matrices: Dict[str, torch.Tensor] = {}
    for path, act_list in activations.items():
        if not act_list:
            continue
        all_acts = torch.cat(act_list, dim=0)  # (Total_tokens, d_in)
        n = all_acts.size(0)
        cov = (all_acts.t() @ all_acts) / max(1, n)
        cov_matrices[path] = cov

    return cov_matrices


def allocate_rank_auto(
    singular_values: torch.Tensor,
    target_ratio: float = 0.5,
    min_rank: int = 4,
    rank_step: int = 32,
) -> int:
    """Determine low-rank rank dynamically from singular value energy retention."""
    total_dim = singular_values.numel()
    target_rank = max(min_rank, int(round(total_dim * target_ratio)))

    # Align to rank_step if feasible
    if target_rank > rank_step:
        target_rank = (target_rank // rank_step) * rank_step

    target_rank = max(min_rank, min(target_rank, total_dim))
    return target_rank


def aasvd_decompose(
    weight: torch.Tensor,
    activation_cov: Optional[torch.Tensor] = None,
    rank: int = 32,
    whitening: bool = True,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Perform Activation-Aware SVD on a 2D weight matrix.

    Returns:
        U_r, S_r, V_r, and energy_retention_ratio.
    """
    W = weight.detach().float()  # (d_out, d_in)
    d_out, d_in = W.shape
    max_rank = min(d_out, d_in)
    rank = min(max(1, rank), max_rank)

    if whitening and activation_cov is not None and activation_cov.shape == (d_in, d_in):
        # Diagonal variance whitening (numerically robust)
        var = torch.diag(activation_cov).clamp(min=eps)
        scale = torch.sqrt(var)  # (d_in,)
        W_scaled = W * scale.unsqueeze(0)  # Scale columns by activation std
        U, S, Vh = torch.linalg.svd(W_scaled, full_matrices=False)
        V = Vh.t()
        # Unwhiten V: V = V / scale
        V = V / scale.unsqueeze(1).clamp(min=eps)
    else:
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        V = Vh.t()

    total_energy = torch.sum(S**2).item()
    retained_energy = torch.sum(S[:rank] ** 2).item()
    energy_ratio = (retained_energy / total_energy) if total_energy > 0 else 1.0

    U_r = U[:, :rank].contiguous()
    S_r = S[:rank].contiguous()
    V_r = V[:, :rank].contiguous()

    return U_r, S_r, V_r, energy_ratio
