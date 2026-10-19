# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
RoPE Compatibility Shim

Backward-compatibility redirection module routing legacy RoPE imports to neuralops.core.utils.rope.
"""

import warnings
warnings.warn(
    "neuralops.core.rope is deprecated. Use neuralops.core.utils.rope.",
    DeprecationWarning,
    stacklevel=2,
)
import torch
from neuralops.core.utils.rope import *  # noqa: F401,F403
from neuralops.core.utils.rope import (  # noqa: F401
    _rotate_half,
    apply_rotary_pos_emb,
)


def apply_rope_after_up_projection(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embeddings (RoPE) AFTER low-rank up-projection to full
    head dimensions, preserving spatial coordinate geometry and relative frequency angles.
    """
    return apply_rotary_pos_emb(q, k, cos, sin, position_ids)


def fused_rope_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) fast path."""
    return apply_rotary_pos_emb(q, k, cos, sin)
