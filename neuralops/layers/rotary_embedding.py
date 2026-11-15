#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Rotary Embedding (rotary_embedding) Module.

Standalone Rotary Embedding class providing precomputed sinusoidal frequency tables,
dynamic sequence expansion, and post-projection head rotation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from neuralops.layers.rope import (
    RoPE,
    RotaryPositionalEmbedding,
    apply_rotary_pos_emb,
    post_projection_rope,
    pre_projection_rope,
)

__all__ = [
    "RoPE",
    "RotaryEmbedding",
    "RotaryPositionalEmbedding",
    "apply_rotary_pos_emb",
    "post_projection_rope",
    "pre_projection_rope",
]


class RotaryEmbedding(RotaryPositionalEmbedding):
    """Rotary Embedding module with configurable base frequencies and scaling factors."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        device: torch.device | None = None,
        **kwargs: Any,  # reason: absorbs unused configuration flags
    ) -> None:
        """Initializes RotaryEmbedding with optional linear frequency scaling."""
        super().__init__(
            dim=dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            device=device,
        )
        self.scaling_factor: float = float(scaling_factor)
        if self.scaling_factor != 1.0:
            self.inv_freq = self.inv_freq / self.scaling_factor
            self._set_cos_sin_cache(max_position_embeddings, device=device, dtype=torch.float32)
