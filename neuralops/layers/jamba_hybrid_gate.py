#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Jamba Hybrid SSM-Attention Gating Layer.

Combines transformer self-attention and Mamba SSM representations via a learned
per-head/per-channel alpha gate:
    out = sigmoid(alpha) * attn_out + (1 - sigmoid(alpha)) * ssm_out
with optional MoE branch contribution.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from neuralops.ops.triton.selective_scan import (
    fused_jamba_hybrid_ssm_attn_gate_pytorch,
    fused_jamba_hybrid_ssm_attn_gate_triton,
)

logger = logging.getLogger(__name__)


class JambaHybridGate(nn.Module):
    """Learned hybrid gating combining attention and SSM outputs.

    Args:
        num_heads: Number of attention heads (or feature dimension for channel-wise gating).
        initial_alpha: Initial gate logit value (default: 0.0 -> 0.5 balanced weighting).
    """

    def __init__(self, num_heads: int, initial_alpha: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.alpha_raw = nn.Parameter(torch.full((num_heads,), float(initial_alpha)))

    def forward(
        self,
        attn_out: torch.Tensor,
        ssm_out: torch.Tensor,
        moe_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies fused hybrid gating between attention, SSM, and optional MoE branches.

        Args:
            attn_out: Attention branch output of shape [B, L, D].
            ssm_out: SSM branch output of shape [B, L, D].
            moe_out: Optional MoE branch output of shape [B, L, D].

        Returns:
            Gated combination tensor of shape [B, L, D].
        """
        if attn_out.is_cuda and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
            try:
                return fused_jamba_hybrid_ssm_attn_gate_triton(attn_out, ssm_out, self.alpha_raw, moe_out)
            except Exception as e:
                # allowed-swallow: best-effort Triton fast-path with robust PyTorch fallback
                logger.debug("Triton Jamba gate failed, falling back to PyTorch: %s", e)

        return fused_jamba_hybrid_ssm_attn_gate_pytorch(attn_out, ssm_out, self.alpha_raw, moe_out)


__all__ = [
    "JambaHybridGate",
    "fused_jamba_hybrid_ssm_attn_gate_pytorch",
    "fused_jamba_hybrid_ssm_attn_gate_triton",
]
