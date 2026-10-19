#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Model State Snapshot and Restorer

Exact bit-identical state preservation and restoration engine for in-memory attention swapping.
"""

from __future__ import annotations

import gc

import torch
import torch.nn as nn

from neuralops.core.model_utils import replace_module


class ModelStateSnapshot:
    """Exact bit-identical snapshot of an nn.Module."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.swapped_layer_paths: list[str] = []
        self.original_submodules: dict[str, nn.Module] = {}
        for name, sub in model.named_modules():
            lname = name.lower()
            if (
                lname.endswith(".attention")
                or lname.endswith(".attn")
                or lname.endswith(".self")
                or lname.endswith(".self_attn")
            ):
                self.swapped_layer_paths.append(name)
                self.original_submodules[name] = sub

        self.clean_state_dict: dict[str, torch.Tensor] = {
            k: v.detach().clone().cpu() for k, v in model.state_dict().items()
        }

        self.was_training = model.training

    def restore(self) -> None:
        """Restores the model structure and tensor values to exact bit-identical snapshot state."""
        for path, original_sub in self.original_submodules.items():
            try:
                replace_module(self.model, path, original_sub)
            except Exception:
                pass

        device = next(self.model.parameters()).device
        restored_state = {k: v.to(device) for k, v in self.clean_state_dict.items()}
        self.model.load_state_dict(restored_state, strict=False)

        self.model.train(self.was_training)

        for attr in ("compression_stats", "is_custom_attention", "_injected_attention"):
            if hasattr(self.model, attr):
                try:
                    delattr(self.model, attr)
                except Exception:
                    pass

        for m in self.model.modules():
            if hasattr(m, "merged"):
                m.merged = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def verify_bit_identical(self) -> tuple[bool, float]:
        """Verify that current model state is bit-identical to snapshot. Returns (is_identical, max_diff)."""
        max_diff = 0.0
        current_st = self.model.state_dict()
        for k, v_orig in self.clean_state_dict.items():
            if k not in current_st:
                return False, float("inf")
            v_curr = current_st[k].detach().cpu()
            if v_orig.shape != v_curr.shape:
                return False, float("inf")
            diff = (v_orig - v_curr).abs().max().item()
            if diff > max_diff:
                max_diff = diff
        return (max_diff == 0.0), max_diff
