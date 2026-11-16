#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Configuration and clamp utilities for GatedHybridKernelAttention."""

from __future__ import annotations

from typing import Any
import torch


def _clamp_to_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Sanitize non-finite values and clamp to dtype safe range before casting."""
    if dtype.is_floating_point:
        max_val = torch.finfo(dtype).max * 0.5
        return torch.nan_to_num(tensor, nan=0.0, posinf=max_val, neginf=-max_val).clamp(-max_val, max_val).to(dtype)
    return torch.nan_to_num(tensor, nan=0.0).to(dtype)


class GatedHybridKernelConfig:
    def __init__(self, **kwargs: Any):
        self.hidden_size = kwargs.get("hidden_size")
        self.num_heads = kwargs.get("num_heads", kwargs.get("num_attention_heads"))
        self.num_attention_heads = self.num_heads
        self.num_key_value_heads = kwargs.get("num_key_value_heads", self.num_heads)
        self.head_dim = kwargs.get("head_dim")
        if self.head_dim is None and self.hidden_size is not None and self.num_heads is not None:
            self.head_dim = self.hidden_size // max(1, self.num_heads)
        self.window_size = kwargs.get("window_size", 128)
        default_svd = min(384, self.hidden_size) if self.hidden_size is not None else 384
        self.svd_rank = kwargs.get("svd_rank", default_svd)
        self.low_rank_update = kwargs.get("low_rank_update", 16)
        self.kernel_feature_dim = kwargs.get("kernel_feature_dim", 128)
        self.dropout = kwargs.get("dropout", 0.1)
        self.use_query_gate = kwargs.get("use_query_gate", True)
        self.use_input_gate = kwargs.get("use_input_gate", True)
        self.use_pre_norm = kwargs.get("use_pre_norm", False)
        self.kernel_method = kwargs.get("kernel_method", "orthogonal")
        self.kernel_clamp_min = kwargs.get("kernel_clamp_min", -20.0)
        self.kernel_clamp_max = kwargs.get("kernel_clamp_max", 8.0)
        self.kernel_seed = kwargs.get("kernel_seed")
        self.fusion_gate_bias = kwargs.get("fusion_gate_bias", 2.5)
        self.is_decoder = kwargs.get("is_decoder", False)
        self.chunk_size = kwargs.get("chunk_size", 128)


AegisConfigV3 = GatedHybridKernelConfig

