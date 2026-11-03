#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
r"""
Hybrid Mixture of Attentions (MoA) Pareto Router.

Allocates full quadratic attention and sub-quadratic linear attention layers
to optimize the accuracy-throughput Pareto frontier across transformer backbones.

Mathematical Optimization:
    Given total layer count L, attention entropy for layer l is defined as:
        \mathcal{H}_l = - \sum_{i, j} A_{i, j}^{(l)} \log (A_{i, j}^{(l)} + \epsilon)
    Boundary layers (e.g. early feature extraction layers 0-3 and late token decoding layers L-2, L-1)
    exhibit high attention entropy and critical associative retrieval demands, retaining Full Attention.
    Intermediate representation-transforming layers (e.g. layers 4 through L-3) exhibit lower entropy
    and are safely routed to Linear Attention.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch

logger = logging.getLogger("neuralops.surgery.moa_router")


class HybridMoARouter:
    """Hybrid Mixture of Attentions (MoA) Pareto Router.

    Partitions transformer layers into Full Attention and Linear Attention regimes
    to maximize inference throughput while preserving generation accuracy.

    Args:
        num_layers: Total number of transformer layers in the backbone (default: 32).
        full_attention_layers: Explicit list of layer indices retaining Full Attention.
            If None, applies the canonical Pareto optimal partition where the first 4
            layers (0, 1, 2, 3) and the last 2 layers (num_layers-2, num_layers-1)
            retain Full Attention, and all intermediate layers use Linear Attention.
    """

    def __init__(
        self,
        num_layers: int = 32,
        full_attention_layers: Optional[List[int]] = None,
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.num_layers = num_layers
        if full_attention_layers is None:
            if num_layers > 6:
                # Default Pareto optimal partition: Layers 0-3 and (L-2)-(L-1) retain Full Attention
                self.full_attention_layers = sorted(
                    list(set([0, 1, 2, 3, num_layers - 2, num_layers - 1]))
                )
            else:
                # For small models, retain boundary layers
                self.full_attention_layers = sorted(
                    list(set([0, max(0, num_layers - 1)]))
                )
        else:
            invalid_layers = [idx for idx in full_attention_layers if idx < 0 or idx >= num_layers]
            if invalid_layers:
                raise ValueError(
                    f"full_attention_layers contains invalid indices outside [0, {num_layers - 1}]: {invalid_layers}"
                )
            self.full_attention_layers = sorted(list(set(full_attention_layers)))

        self._full_set = set(self.full_attention_layers)
        self._linear_layers = [i for i in range(self.num_layers) if i not in self._full_set]

    def get_layer_type(self, layer_idx: int) -> Literal["full_attention", "linear_attention"]:
        """Returns the attention paradigm allocated for a specific layer index."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"Layer index {layer_idx} out of range for model with {self.num_layers} layers."
            )
        if layer_idx in self._full_set:
            return "full_attention"
        return "linear_attention"

    def is_full_attention(self, layer_idx: int) -> bool:
        """Returns True if the layer should retain standard quadratic Full Attention."""
        return layer_idx in self._full_set

    def is_linear_attention(self, layer_idx: int) -> bool:
        """Returns True if the layer should be replaced with Linear Attention."""
        return layer_idx not in self._full_set

    def get_full_attention_layers(self) -> List[int]:
        """Returns the sorted list of layer indices retaining Full Attention."""
        return list(self.full_attention_layers)

    def get_linear_attention_layers(self) -> List[int]:
        """Returns the sorted list of layer indices targeted for Linear Attention surgery."""
        return list(self._linear_layers)

    def partition_schedule(self) -> Dict[int, str]:
        """Returns complete layer-by-layer attention allocation schedule."""
        return {layer: self.get_layer_type(layer) for layer in range(self.num_layers)}

    @classmethod
    def compute_entropy_partition(
        cls,
        attention_matrices: Sequence[torch.Tensor],
        num_full_layers: int = 6,
        tau_entropy: Optional[float] = None,
        eps: float = 1e-9,
    ) -> HybridMoARouter:
        r"""Dynamically computes Pareto optimal layer allocation using attention entropy profiles.

        For each layer l:
            \mathcal{H}_l = - \frac{1}{B \cdot H \cdot T} \sum_{b, h, i, j} A_{b, h, i, j} \log(A_{b, h, i, j} + \epsilon)

        Layers with highest entropy or entropy above tau_entropy retain Full Attention.

        Args:
            attention_matrices: List of attention probability tensors [B, H, T, T] for each layer.
            num_full_layers: Target number of layers to retain Full Attention if tau_entropy is None.
            tau_entropy: Optional explicit entropy cutoff threshold.
            eps: Epsilon to prevent log(0).

        Returns:
            Configured HybridMoARouter instance.
        """
        num_layers = len(attention_matrices)
        if num_layers == 0:
            raise ValueError("attention_matrices sequence cannot be empty")

        entropies: List[Tuple[int, float]] = []
        for l_idx, attn in enumerate(attention_matrices):
            with torch.no_grad():
                probs = attn.to(torch.float32).clamp(min=eps)
                # Compute normalized entropy
                entropy = -(probs * torch.log(probs)).sum(dim=-1).mean().item()
                entropies.append((l_idx, entropy))

        if tau_entropy is not None:
            full_layers = [l_idx for l_idx, ent in entropies if ent >= tau_entropy]
            if not full_layers:
                # Always retain at least the first layer if none cross threshold
                full_layers = [0]
        else:
            # Sort descending by entropy
            sorted_by_entropy = sorted(entropies, key=lambda x: x[1], reverse=True)
            k = min(num_full_layers, num_layers)
            full_layers = sorted([x[0] for x in sorted_by_entropy[:k]])

        logger.info(
            "Computed dynamic MoA entropy partition: %d full attention layers %s across %d total layers.",
            len(full_layers),
            full_layers,
            num_layers,
        )
        return cls(num_layers=num_layers, full_attention_layers=full_layers)

    def summary(self) -> str:
        """Returns human-readable summary of the MoA partition schedule."""
        full_cnt = len(self.full_attention_layers)
        linear_cnt = len(self._linear_layers)
        ratio = (linear_cnt / self.num_layers) * 100.0 if self.num_layers > 0 else 0.0
        return (
            f"HybridMoARouter(total_layers={self.num_layers}, "
            f"full_attention={full_cnt} {self.full_attention_layers}, "
            f"linear_attention={linear_cnt} [{linear_cnt} layers, {ratio:.1f}%])"
        )


__all__ = [
    "HybridMoARouter",
]
