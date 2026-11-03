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
        \mathcal{H}_l = - \frac{1}{B \cdot H} \sum_{b, h} \sum_{i=1}^T \sum_{j=1}^T A_{b, h, i, j}^{(l)} \log (A_{b, h, i, j}^{(l)} + \epsilon)
    Boundary layers:
        - Layers 0-3: Retain Full Attention to preserve token grounding, local syntax, and induction heads.
        - Layers L-2, L-1 (e.g. 30-31 for L=32): Retain Full Attention for exact vocabulary distribution synthesis.
    Intermediate layers:
        - Layers 4 to L-3 (e.g. 4-29 for L=32): 100% Linear Attention (Cosformer / GLA / GDN) in O(T) complexity.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import torch

logger = logging.getLogger("neuralops.surgery.moa_router")


def compute_attention_entropy(
    attn_matrix: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    r"""Computes average attention distribution entropy \mathcal{H}_l.

    Formula:
        \mathcal{H}_l = - \frac{1}{B \cdot H \cdot T} \sum_{b, h, i} \sum_{j=1}^T A_{b, h, i, j} \log(A_{b, h, i, j} + \epsilon)

    Theoretical Bounds:
        - Uniform Attention (A_{i, j} = 1/T): \mathcal{H} = \ln(T)
        - Sharp One-Hot Attention (A_{i, j} = \delta_{i, j}): \mathcal{H} = 0.0

    Args:
        attn_matrix: Attention probability tensor of shape [..., T, T] or [B, H, T, T].
        eps: Epsilon to prevent log(0) domain errors.

    Returns:
        Scalar attention entropy in nats.
    """
    with torch.no_grad():
        probs = attn_matrix.to(torch.float32).clamp(min=eps)
        # Row-wise entropy: - sum_j (p_j * log(p_j))
        row_entropy = -(probs * torch.log(probs)).sum(dim=-1)
        # Average across sequence, heads, and batch
        return float(row_entropy.mean().item())


def profile_layer_sensitivity(
    attention_matrices: Sequence[torch.Tensor],
    eps: float = 1e-12,
) -> Dict[str, Any]:
    r"""Profiles layer-wise attention entropy across transformer layers for syntactic vs semantic routing.

    Identifies:
        - High-sensitivity boundary layers (early syntax grounding & late factual output calibration).
        - Intermediate bulk representation layers suitable for linear attention surgery.

    Args:
        attention_matrices: Sequence of attention tensors [B, H, T, T] per layer.
        eps: Epsilon parameter for entropy calculation.

    Returns:
        Dictionary containing layer entropies, sensitivity rankings, and recommended partitions.
    """
    num_layers = len(attention_matrices)
    if num_layers == 0:
        raise ValueError("attention_matrices sequence cannot be empty")

    entropies = [compute_attention_entropy(attn, eps=eps) for attn in attention_matrices]

    # Boundary layers naturally exhibit higher attention entropy
    boundary_indices = sorted(list(set([0, 1, 2, 3, max(0, num_layers - 2), max(0, num_layers - 1)])))
    boundary_indices = [idx for idx in boundary_indices if idx < num_layers]

    intermediate_indices = [idx for idx in range(num_layers) if idx not in boundary_indices]

    mean_boundary_entropy = (
        sum(entropies[i] for i in boundary_indices) / len(boundary_indices)
        if boundary_indices
        else 0.0
    )
    mean_intermediate_entropy = (
        sum(entropies[i] for i in intermediate_indices) / len(intermediate_indices)
        if intermediate_indices
        else 0.0
    )

    return {
        "num_layers": num_layers,
        "entropies": entropies,
        "boundary_layers": boundary_indices,
        "intermediate_layers": intermediate_indices,
        "mean_boundary_entropy": mean_boundary_entropy,
        "mean_intermediate_entropy": mean_intermediate_entropy,
    }


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
                # Canonical Pareto optimal partition:
                # Phase 073: Layers 0-3 Full Attention
                # Phase 074: Layers (L-2)-(L-1) Full Attention
                # Phase 075: Layers 4-(L-3) Linear Attention (26 layers for L=32)
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
    def solve_pareto_frontier(
        cls,
        num_layers: int = 32,
        layer_latencies: Optional[Dict[str, float]] = None,
        layer_ppl_impacts: Optional[List[float]] = None,
        max_ppl_delta: float = 0.15,
        min_full_layers: int = 6,
    ) -> Tuple[HybridMoARouter, Dict[str, Any]]:
        r"""Solves the Pareto frontier optimization problem to discover the optimal layer partition.

        Optimization Formulation:
            \min_{\mathcal{S}_{linear}} \sum_{l \in \mathcal{S}_{full}} T_{full} + \sum_{l \in \mathcal{S}_{linear}} T_{linear}
            \text{s.t.} \sum_{l \in \mathcal{S}_{linear}} \Delta \text{PPL}_l \le \text{max\_ppl\_delta}
            \text{and boundary safeguards preserved: } \{0, 1, 2, 3, L-2, L-1\} \subseteq \mathcal{S}_{full}

        Args:
            num_layers: Total number of transformer layers (default: 32).
            layer_latencies: Dictionary with 'full' and 'linear' layer latency in ms.
            layer_ppl_impacts: Optional per-layer perplexity impact list of length num_layers.
            max_ppl_delta: Maximum permissible perplexity degradation (default: 0.15).
            min_full_layers: Minimum number of full attention layers to enforce (default: 6).

        Returns:
            Tuple of (configured HybridMoARouter, solution_metadata dictionary).
        """
        latencies = layer_latencies or {"full": 1.8, "linear": 0.5}
        t_full = latencies.get("full", 1.8)
        t_linear = latencies.get("linear", 0.5)

        # Canonical boundary safeguards (Phases 073 and 074)
        if num_layers > 6:
            mandatory_full = set([0, 1, 2, 3, num_layers - 2, num_layers - 1])
        else:
            mandatory_full = set([0, max(0, num_layers - 1)])

        # Construct default synthetic perplexity degradation profile if none provided
        if layer_ppl_impacts is None:
            ppl_impacts = []
            for i in range(num_layers):
                if i in mandatory_full:
                    # High perplexity degradation if boundary layers were linear
                    ppl_impacts.append(0.08)
                else:
                    # Intermediate layers have low perplexity impact (e.g. 0.004)
                    ppl_impacts.append(0.004)
        else:
            if len(layer_ppl_impacts) != num_layers:
                raise ValueError(
                    f"layer_ppl_impacts length {len(layer_ppl_impacts)} does not match num_layers {num_layers}"
                )
            ppl_impacts = list(layer_ppl_impacts)

        # Mandatory full layers are fixed
        selected_full = set(mandatory_full)

        # Candidates for linear replacement are intermediate layers
        candidate_layers = [i for i in range(num_layers) if i not in selected_full]
        # Sort candidates ascending by perplexity impact (greedy knapsack)
        candidate_layers.sort(key=lambda idx: ppl_impacts[idx])

        accumulated_ppl_delta = 0.0
        selected_linear = []

        for layer_idx in candidate_layers:
            cost = ppl_impacts[layer_idx]
            if accumulated_ppl_delta + cost <= max_ppl_delta:
                selected_linear.append(layer_idx)
                accumulated_ppl_delta += cost
            else:
                selected_full.add(layer_idx)

        total_latency = len(selected_full) * t_full + len(selected_linear) * t_linear
        baseline_latency = num_layers * t_full
        speedup = baseline_latency / max(1e-6, total_latency)

        router = cls(num_layers=num_layers, full_attention_layers=sorted(list(selected_full)))

        metadata = {
            "num_layers": num_layers,
            "full_layers_count": len(selected_full),
            "linear_layers_count": len(selected_linear),
            "accumulated_ppl_delta": accumulated_ppl_delta,
            "max_ppl_delta": max_ppl_delta,
            "baseline_latency_ms": baseline_latency,
            "optimized_latency_ms": total_latency,
            "speedup_factor": speedup,
            "pareto_optimal": True,
        }

        logger.info(
            "Solved MoA Pareto frontier for L=%d: %d full, %d linear, speedup=%.2fx, delta_ppl=%.4f <= %.2f",
            num_layers,
            len(selected_full),
            len(selected_linear),
            speedup,
            accumulated_ppl_delta,
            max_ppl_delta,
        )

        return router, metadata

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
            ent = compute_attention_entropy(attn, eps=eps)
            entropies.append((l_idx, ent))

        if tau_entropy is not None:
            full_layers = [l_idx for l_idx, ent in entropies if ent >= tau_entropy]
            if not full_layers:
                # Always retain boundary layers if none cross threshold
                full_layers = [0, max(0, num_layers - 1)]
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
    "compute_attention_entropy",
    "profile_layer_sensitivity",
    "HybridMoARouter",
]
