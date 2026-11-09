#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Wanda: Pruning by Weights and Activations

Prunes weights according to the product of weight magnitudes and input activation norms per output channel.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from neuralops.core.base import BaseCompressor
from neuralops.core.common import _cfg_get
from neuralops.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["WandaCompressor", "wanda_prune_layer", "wanda_prune_model", "wanda_factory"]


def _pytorch_wanda(
    W: torch.Tensor,
    sparsity: float = 0.5,
    act_norm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized PyTorch reference implementation for Wanda pruning (Level 3 fallback)."""
    if act_norm is None:
        act_norm = torch.ones(W.shape[1], device=W.device, dtype=W.dtype)
    metric = torch.abs(W) * act_norm.unsqueeze(0)
    n_keep = max(1, min(metric.shape[1], int(round(metric.shape[1] * (1.0 - sparsity)))))
    _, top_idx = torch.topk(metric, k=n_keep, dim=1, largest=True)
    mask = torch.zeros_like(W, dtype=torch.bool)
    mask.scatter_(1, top_idx, True)
    return W * mask.to(W.dtype)


def wanda_prune_layer(
    module: nn.Linear,
    sparsity: float = 0.5,
    calib_data: torch.Tensor | list[torch.Tensor] | None = None,
    act_norm: torch.Tensor | None = None,
) -> tuple[nn.Linear, torch.Tensor]:
    """Prunes linear layer weights based on the Wanda importance score S_ij = |W_ij| * ||X_j||_2.

    Dispatches across 3-tier hierarchy: CUDA C++ (Level 1) -> Triton JIT (Level 2) -> PyTorch (Level 3).

    Args:
        module: The target nn.Linear module.
        sparsity: Desired sparsity level in range [0.0, 1.0).
        calib_data: Calibration activations tensor or list of batch tensors.
        act_norm: Precomputed L2 norm vector across input features.

    Returns:
        Tuple of (pruned module, boolean mask of kept weights).
    """
    if sparsity <= 0.0:
        mask = torch.ones_like(module.weight.data, dtype=torch.bool)
        return module, mask

    W = module.weight.data

    if act_norm is None:
        if calib_data is None:
            act_norm = torch.ones(W.shape[1], device=W.device, dtype=W.dtype)
        else:
            if isinstance(calib_data, list):
                stacked = torch.cat([x.reshape(-1, x.shape[-1]) for x in calib_data], dim=0)
            else:
                stacked = calib_data.reshape(-1, calib_data.shape[-1])
            if stacked.shape[-1] != W.shape[1]:
                act_norm = torch.ones(W.shape[1], device=W.device, dtype=W.dtype)
            else:
                act_norm = torch.norm(stacked.to(torch.float32), p=2, dim=0).to(W.device, dtype=W.dtype)

    try:
        from neuralops.ops.cuda._fallback import dispatch_wanda

        W_pruned = dispatch_wanda(W, sparsity=sparsity, act_norm=act_norm, pytorch_fn=_pytorch_wanda)
        mask = W_pruned != 0.0
    except Exception as exc:
        logger.debug("dispatch_wanda fallback to PyTorch reference: %s", exc)
        W_pruned = _pytorch_wanda(W, sparsity=sparsity, act_norm=act_norm)
        mask = W_pruned != 0.0

    module.weight.data.copy_(W_pruned)
    module.register_buffer("pruning_mask", mask)
    module.register_buffer("mask", mask.to(dtype=module.weight.dtype))
    return module, mask


def wanda_prune_model(
    model: nn.Module,
    calib_data: list[torch.Tensor] | None = None,
    sparsity: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Applies Wanda pruning to all eligible linear layers of model, protecting task heads."""
    from neuralops.utils.compat.architectures import should_compress_linear

    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if not should_compress_linear(model, name):
                continue
            _, mask = wanda_prune_layer(module, sparsity=sparsity, calib_data=calib_data)
            masks[name] = mask
    return masks


@COMPRESSOR_REGISTRY.register("wanda")
@COMPRESSOR_REGISTRY.register("wanda_pruner", is_alias=True)
@COMPRESSOR_REGISTRY.register("wanda_compressor", is_alias=True)
class WandaCompressor(BaseCompressor):
    """Wanda Compressor: Pruning by Weights and Activations.

    Prunes weights according to the product of weight magnitudes and input activation norms.
    """

    FAMILY: str = "pruning"

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        calib_data: list[torch.Tensor] | None = None,
        comp_config: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="wanda", **kwargs)
        self.sparsity = _cfg_get(comp_config, "wanda_sparsity", _cfg_get(comp_config, "sparsity", sparsity))
        self.calib_data = calib_data
        self.masks: dict[str, torch.Tensor] = {}

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        self.masks = wanda_prune_model(
            self.model,
            calib_data=self.calib_data,
            sparsity=self.sparsity,
        )
        self.record_compression_check(status="success")
        stats = self.get_compression_stats()
        stats["num_pruned_layers"] = len(self.masks)
        stats["sparsity"] = self.sparsity
        self.stats = stats
        return self.model, stats


def wanda_factory(
    model: nn.Module,
    comp_config: Any = None,
    sparsity: float = 0.5,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> nn.Module | tuple[nn.Module, dict[str, Any]]:
    """Factory function for Wanda compressor."""
    ret_stats = return_stats or return_tuple or kwargs.pop("return_both", False)
    compressor = WandaCompressor(model, comp_config=comp_config, sparsity=sparsity, **kwargs)
    compressed_model, stats = compressor.compress()
    if ret_stats:
        return compressed_model, stats
    return compressed_model
