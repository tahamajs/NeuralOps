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

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

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


class AttentionEntropyProfile(dict):
    r"""Telemetry profile of layer-wise attention entropy and routing recommendations.

    Inherits from dict to maintain transparent backwards compatibility with dictionary
    subscripting (`profile['num_layers']`), while providing object attribute access.
    """

    def __init__(
        self,
        num_layers: int,
        entropies: Sequence[float],
        boundary_layers: Sequence[int],
        intermediate_layers: Sequence[int],
        mean_boundary_entropy: float,
        mean_intermediate_entropy: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        data: Dict[str, Any] = {
            "num_layers": num_layers,
            "entropies": list(entropies),
            "boundary_layers": list(boundary_layers),
            "intermediate_layers": list(intermediate_layers),
            "mean_boundary_entropy": float(mean_boundary_entropy),
            "mean_intermediate_entropy": float(mean_intermediate_entropy),
        }
        if metadata:
            data["metadata"] = metadata
        super().__init__(data)
        self.num_layers = num_layers
        self.entropies = list(entropies)
        self.boundary_layers = list(boundary_layers)
        self.intermediate_layers = list(intermediate_layers)
        self.mean_boundary_entropy = float(mean_boundary_entropy)
        self.mean_intermediate_entropy = float(mean_intermediate_entropy)
        self.metadata = metadata or {}


def profile_layer_sensitivity(
    attention_matrices: Sequence[torch.Tensor],
    eps: float = 1e-12,
) -> AttentionEntropyProfile:
    r"""Profiles layer-wise attention entropy across transformer layers for syntactic vs semantic routing.

    Identifies:
        - High-sensitivity boundary layers (early syntax grounding & late factual output calibration).
        - Intermediate bulk representation layers suitable for linear attention surgery.

    Args:
        attention_matrices: Sequence of attention tensors [B, H, T, T] per layer.
        eps: Epsilon parameter for entropy calculation.

    Returns:
        AttentionEntropyProfile containing layer entropies, sensitivity rankings, and recommended partitions.
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

    return AttentionEntropyProfile(
        num_layers=num_layers,
        entropies=entropies,
        boundary_layers=boundary_indices,
        intermediate_layers=intermediate_indices,
        mean_boundary_entropy=mean_boundary_entropy,
        mean_intermediate_entropy=mean_intermediate_entropy,
    )


class MiddleLayersRoutingResult(list):
    """Result of middle bulk linear replacement routing.

    Subclasses list so that inspecting or asserting against the returned object
    behaves identically to a list of linear layer indices, while providing full
    delegation to the underlying MoARouter.
    """

    def __init__(
        self,
        linear_layers: Sequence[int],
        router: MoARouter,
        full_layers: Sequence[int],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(sorted(linear_layers))
        self.router = router
        self.linear_layers = sorted(list(linear_layers))
        self.full_attention_layers = sorted(list(full_layers))
        self.metadata = metadata or {}

    def to_router(self) -> MoARouter:
        """Returns the underlying configured MoARouter."""
        return self.router

    def partition_schedule(self) -> Dict[int, str]:
        """Returns the partition schedule dictionary."""
        return self.router.partition_schedule()

    def is_linear_attention(self, layer_idx: int) -> bool:
        """Checks if layer is linear attention."""
        return self.router.is_linear_attention(layer_idx)

    def is_full_attention(self, layer_idx: int) -> bool:
        """Checks if layer is full attention."""
        return self.router.is_full_attention(layer_idx)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes router configuration to dictionary."""
        return self.router.to_dict()


class ParetoAssignmentVector(list):
    r"""Optimal layer-type assignment vector \mathbf{z}^* \in \{0, 1\}^L.

    Subclasses list so indexing `z[l]` returns 0 (full) or 1 (linear),
    while exposing helper methods, metadata, and router conversions.
    """

    def __init__(
        self,
        assignment: Sequence[int],
        router: MoARouter,
        metadata: Dict[str, Any],
    ) -> None:
        super().__init__(assignment)
        self.vector = list(assignment)
        self.router = router
        self.metadata = metadata
        self.linear_layers = [i for i, z in enumerate(assignment) if z == 1]
        self.full_attention_layers = [i for i, z in enumerate(assignment) if z == 0]

    def to_router(self) -> MoARouter:
        """Returns the configured MoARouter instance."""
        return self.router

    def to_tensor(self, dtype: torch.dtype = torch.int64) -> torch.Tensor:
        """Converts assignment vector to a 1D PyTorch tensor."""
        return torch.tensor(self.vector, dtype=dtype)


class CanonicalSchedule(dict):
    """Canonical layer-by-layer attention allocation schedule.

    Subclasses dict so `schedule[l]` returns 'full_attention' or 'linear_attention',
    while exposing router conversion and layer partitions.
    """

    def __init__(
        self,
        depth: int,
        full_attention_layers: Sequence[int],
        linear_attention_layers: Sequence[int],
        schedule_map: Dict[int, str],
        router: MoARouter,
    ) -> None:
        super().__init__(schedule_map)
        self.depth = depth
        self.num_layers = depth
        self.full_attention_layers = sorted(list(full_attention_layers))
        self.linear_attention_layers = sorted(list(linear_attention_layers))
        self.assignment_vector = [1 if i in self.linear_attention_layers else 0 for i in range(depth)]
        self.router = router

    def to_router(self) -> MoARouter:
        """Returns the configured MoARouter instance."""
        return self.router


CANONICAL_SCHEDULE_PARTITIONS: Dict[int, Dict[str, List[int]]] = {
    12: {
        "full_attention_layers": [0, 1, 2, 3, 10, 11],
        "linear_attention_layers": list(range(4, 10)),  # [4, 5, 6, 7, 8, 9] (6 layers)
    },
    24: {
        "full_attention_layers": [0, 1, 2, 3, 22, 23],
        "linear_attention_layers": list(range(4, 22)),  # [4, ..., 21] (18 layers)
    },
    32: {
        "full_attention_layers": [0, 1, 2, 3, 30, 31],
        "linear_attention_layers": list(range(4, 30)),  # [4, ..., 29] (26 layers)
    },
    48: {
        "full_attention_layers": [0, 1, 2, 3, 46, 47],
        "linear_attention_layers": list(range(4, 46)),  # [4, ..., 45] (42 layers)
    },
}


class class_or_instance_method:
    """Descriptor enabling a method to be invoked cleanly on both classes and instances."""

    def __init__(self, fn: Any) -> None:
        self.fn = fn

    def __get__(self, instance: Any, owner: Any) -> Any:
        if instance is None:
            return lambda *args, **kwargs: self.fn(owner, *args, **kwargs)
        return lambda *args, **kwargs: self.fn(instance, *args, **kwargs)


class MoARouter:
    """Hybrid Mixture of Attentions (MoA) Pareto Router.

    Partitions transformer layers into Full Attention and Linear Attention regimes
    to maximize inference throughput while preserving generation accuracy.

    Args:
        num_layers: Total number of transformer layers in the backbone (default: 32).
        full_attention_layers: Explicit list of layer indices retaining Full Attention.
            If None, applies the canonical Pareto optimal partition where the first 4
            layers (0, 1, 2, 3) and the last 2 layers (num_layers-2, num_layers-1)
            retain Full Attention, and all intermediate layers use Linear Attention.
        entropies: Optional sequence of layer-wise attention entropy measurements.
    """

    def __init__(
        self,
        num_layers: int = 32,
        full_attention_layers: Optional[List[int]] = None,
        entropies: Optional[Sequence[float]] = None,
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.num_layers = num_layers
        if full_attention_layers is None:
            if num_layers > 6:
                # Canonical Pareto optimal partition:
                # Boundary safeguards: Layers 0-3 and (L-2)-(L-1) Full Attention
                # Intermediate bulk: Layers 4-(L-3) Linear Attention (26 layers for L=32)
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
        self._entropies: List[float] = list(entropies) if entropies is not None else []
        self.telemetry_profile: Dict[int, Dict[str, Any]] = {}

    def get_boundary_layers(self) -> List[int]:
        """Returns the sorted list of protected boundary layer indices (Layers 0-3 and L-2..L-1)."""
        if self.num_layers > 6:
            return sorted(list(set([0, 1, 2, 3, self.num_layers - 2, self.num_layers - 1])))
        return sorted(list(set([0, max(0, self.num_layers - 1)])))

    def is_boundary_layer(self, layer_idx: int) -> bool:
        """Returns True if the given layer index is a protected boundary layer."""
        return layer_idx in set(self.get_boundary_layers())

    def compute_layer_entropy(
        self_or_attn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        r"""Computes attention distribution entropy \mathcal{H}_l in bits (base 2).

        Mathematical Formulation:
            \mathcal{H}_l = - \frac{1}{B T H} \sum_{b, t, h, s} A_{l, b, h, t, s} \log_2 (A_{l, b, h, t, s} + \epsilon)

        Interpretation:
            - High entropy (diffuse attention): Attention is spread broadly across sequence,
              making the layer suitable for linear attention surgery (Cosformer / GLA / GDN).
            - Low entropy (sharp retrieval): Attention is focused on key anchor tokens
              (e.g. induction heads, vocabulary calibration), requiring Full Attention or MLA.

        DoD Gate:
            Layer entropy scores computed and recorded in structured telemetry profile.

        Args:
            self_or_attn: MoARouter instance or attention matrix tensor.
            *args: Additional positional arguments (attn_matrix, layer_idx, eps).
            **kwargs: Keyword arguments (attn_matrix, layer_idx, eps).

        Returns:
            Scalar layer entropy \mathcal{H}_l in bits.
        """
        router_inst: Optional[MoARouter] = None
        if isinstance(self_or_attn, torch.Tensor):
            attn_matrix = self_or_attn
            layer_idx = args[0] if len(args) > 0 else kwargs.get("layer_idx", None)
            eps = args[1] if len(args) > 1 else kwargs.get("eps", 1e-12)
            router_inst = kwargs.get("router", None)
        elif isinstance(self_or_attn, MoARouter):
            router_inst = self_or_attn
            attn_matrix = args[0] if len(args) > 0 else kwargs.get("attn_matrix", None)
            layer_idx = args[1] if len(args) > 1 else kwargs.get("layer_idx", None)
            eps = args[2] if len(args) > 2 else kwargs.get("eps", 1e-12)
        else:
            raise TypeError(
                f"Invalid first argument to compute_layer_entropy: expected MoARouter or torch.Tensor, got {type(self_or_attn).__name__}"
            )

        if attn_matrix is None or not isinstance(attn_matrix, torch.Tensor):
            raise ValueError("compute_layer_entropy requires a valid torch.Tensor attn_matrix")

        with torch.no_grad():
            probs = attn_matrix.to(torch.float32).clamp(min=eps)
            row_entropy = -(probs * torch.log2(probs)).sum(dim=-1)
            entropy_val = float(row_entropy.mean().item())

        seq_len = attn_matrix.shape[-2] if attn_matrix.ndim >= 2 else 1
        num_heads = attn_matrix.shape[-3] if attn_matrix.ndim >= 3 else 1
        max_possible_entropy = math.log2(max(2, seq_len))
        diffuse_score = min(1.0, max(0.0, entropy_val / max(1e-6, max_possible_entropy)))
        recommended = "linear_attention" if diffuse_score >= 0.65 else "full_attention"

        if layer_idx is not None:
            actual_idx = int(layer_idx)
            profile_entry = {
                "layer_idx": actual_idx,
                "entropy_bits": entropy_val,
                "diffuse_score": diffuse_score,
                "recommended_paradigm": recommended,
                "seq_len": seq_len,
                "num_heads": num_heads,
                "max_possible_entropy_bits": max_possible_entropy,
            }
            if router_inst is not None:
                router_inst.telemetry_profile[actual_idx] = profile_entry
                if len(router_inst._entropies) <= actual_idx:
                    router_inst._entropies.extend([0.0] * (actual_idx + 1 - len(router_inst._entropies)))
                router_inst._entropies[actual_idx] = entropy_val

        return entropy_val

    def generate_schedule(
        self,
        override_layers: Optional[Dict[int, str]] = None,
        default_intermediate: str = "linear_attention",
        boundary_paradigm: str = "full_attention",
        allow_mla_boundary: bool = True,
    ) -> Dict[int, str]:
        r"""Generates complete layer-by-layer attention schedule enforcing boundary safeguards.

        Inviolable Routing Rule:
            - Layers 0, 1, 2, 3 (early syntax, token grounding, induction heads)
            - Layers L-2, L-1 (output token decoding and vocabulary distribution calibration)
            MUST be locked to 'full_attention' or 'mla' to preserve linguistic coherence.

        DoD Gate:
            Boundary layers strictly protected by automated assertion.

        Args:
            override_layers: Optional layer-to-paradigm overrides.
            default_intermediate: Default paradigm for intermediate layers (default: 'linear_attention').
            boundary_paradigm: Default paradigm for boundary layers (default: 'full_attention').
            allow_mla_boundary: Whether 'mla' is permitted on boundary layers (default: True).

        Returns:
            Dictionary mapping layer index (0 to num_layers - 1) to attention paradigm.

        Raises:
            AssertionError: If any boundary layer is scheduled to anything other than
                'full_attention' or 'mla'.
        """
        boundary_layers = self.get_boundary_layers()
        valid_boundary = ("full_attention", "mla", "full") if allow_mla_boundary else ("full_attention", "full")

        schedule: Dict[int, str] = {}
        for l in range(self.num_layers):
            if l in boundary_layers:
                schedule[l] = boundary_paradigm
            else:
                schedule[l] = default_intermediate

        if override_layers is not None:
            for l_idx, paradigm in override_layers.items():
                if 0 <= l_idx < self.num_layers:
                    schedule[l_idx] = paradigm

        # Phase 070 DoD Gate: Inviolable Boundary Layer Assertion Guard
        for b_idx in boundary_layers:
            assigned = schedule[b_idx]
            assert assigned in valid_boundary, (
                f"Inviolable boundary layer safeguard violated at layer {b_idx}: "
                f"assigned '{assigned}', but boundary layers {boundary_layers} "
                f"must be strictly locked to 'full_attention' or 'mla' to preserve linguistic coherence."
            )

        return schedule

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

    # =========================================================================
    # Phase 071: Middle Bulk Linear Replacement Routing (Layers 4 to L-3)
    # =========================================================================
    @class_or_instance_method
    def route_middle_layers(
        cls_or_self,
        entropies: Optional[Sequence[float]] = None,
        k_linear: Optional[int] = None,
        num_layers: Optional[int] = None,
        boundary_prefix: int = 4,
        boundary_suffix: int = 2,
    ) -> MiddleLayersRoutingResult:
        r"""Routes middle layers to linear attention based on descending entropy order."""
        if isinstance(cls_or_self, HybridMoARouter):
            router = cls_or_self
            l_total = num_layers if num_layers is not None else router.num_layers
        else:
            if num_layers is not None:
                l_total = num_layers
            elif entropies is not None:
                l_total = len(entropies)
            else:
                l_total = 32
            router = cls_or_self(num_layers=l_total)

        return cls_or_self._do_route_middle_layers(
            router=router,
            l_total=l_total,
            entropies=entropies,
            k_linear=k_linear,
            boundary_prefix=boundary_prefix,
            boundary_suffix=boundary_suffix,
        )

    @classmethod
    def route_middle_layers_static(
        cls,
        entropies: Optional[Sequence[float]] = None,
        k_linear: Optional[int] = None,
        num_layers: Optional[int] = None,
        boundary_prefix: int = 4,
        boundary_suffix: int = 2,
    ) -> MiddleLayersRoutingResult:
        if num_layers is not None:
            l_total = num_layers
        elif entropies is not None:
            l_total = len(entropies)
        else:
            l_total = 32
        router = cls(num_layers=l_total)
        return cls._do_route_middle_layers(
            router=router,
            l_total=l_total,
            entropies=entropies,
            k_linear=k_linear,
            boundary_prefix=boundary_prefix,
            boundary_suffix=boundary_suffix,
        )

    @staticmethod
    def _do_route_middle_layers(
        router: MoARouter,
        l_total: int,
        entropies: Optional[Sequence[float]] = None,
        k_linear: Optional[int] = None,
        boundary_prefix: int = 4,
        boundary_suffix: int = 2,
    ) -> MiddleLayersRoutingResult:

        if entropies is not None:
            if len(entropies) != l_total:
                raise ValueError(
                    f"entropies length ({len(entropies)}) does not match num_layers ({l_total})"
                )
            router._entropies = [float(e) for e in entropies]

        # Determine boundary and intermediate layers
        if l_total > (boundary_prefix + boundary_suffix):
            early_bound = set(range(boundary_prefix))
            late_bound = set(range(l_total - boundary_suffix, l_total))
            middle_candidates = [i for i in range(l_total) if i not in early_bound and i not in late_bound]
        else:
            early_bound = {0}
            late_bound = {max(0, l_total - 1)}
            middle_candidates = [i for i in range(l_total) if i not in early_bound and i not in late_bound]

        current_entropies = entropies if entropies is not None else router._entropies

        if current_entropies and len(current_entropies) == l_total:
            # Sort candidate middle layers in descending order of attention entropy
            sorted_middle = sorted(
                middle_candidates,
                key=lambda idx: current_entropies[idx],
                reverse=True,
            )
            if k_linear is not None:
                k = min(max(0, k_linear), len(middle_candidates))
                selected_linear = set(sorted_middle[:k])
            else:
                selected_linear = set(middle_candidates)
        else:
            if k_linear is not None:
                k = min(max(0, k_linear), len(middle_candidates))
                selected_linear = set(middle_candidates[:k])
            else:
                selected_linear = set(middle_candidates)

        selected_full = sorted([l for l in range(l_total) if l not in selected_linear])
        selected_linear_list = sorted(list(selected_linear))

        router.full_attention_layers = selected_full
        router._full_set = set(selected_full)
        router._linear_layers = selected_linear_list

        logger.info(
            "Phase 071: Routed middle bulk layers for L=%d: %d linear %s, %d full %s",
            l_total,
            len(selected_linear_list),
            selected_linear_list,
            len(selected_full),
            selected_full,
        )

        return MiddleLayersRoutingResult(
            linear_layers=selected_linear_list,
            router=router,
            full_layers=selected_full,
        )

    # =========================================================================
    # Phase 072: Multi-Objective Pareto Frontier Solver
    # =========================================================================
    @class_or_instance_method
    def solve_pareto_schedule(
        cls_or_self,
        num_layers: Optional[int] = None,
        max_ppl_delta: float = 0.15,
        layer_latencies: Optional[Dict[str, float]] = None,
        layer_ppl_impacts: Optional[List[float]] = None,
        layer_speedups: Optional[List[float]] = None,
        min_full_layers: int = 6,
    ) -> ParetoAssignmentVector:
        r"""Solves binary integer linear program (ILP) for Pareto optimal layer schedule.

        Mathematical Optimization Formulation:
            \max_{\mathbf{z} \in \{0, 1\}^L} \sum_{l=0}^{L-1} z_l \cdot \text{Speedup}_l
            \text{subject to:}
                \sum_{l=0}^{L-1} z_l \cdot \Delta \text{PPL}_l \le \text{max\_ppl\_delta}
                z_l = 0 \text{ for } l \in \mathcal{B} = \{0, 1, 2, 3, L-2, L-1\}
                \sum_{l=0}^{L-1} (1 - z_l) \ge \text{min\_full\_layers}
                z_l \in \{0, 1\}

        Args:
            num_layers: Total number of transformer layers (default: 32).
            max_ppl_delta: Maximum permissible perplexity degradation budget (default: 0.15).
            layer_latencies: Latency lookup dict with 'full' and 'linear' in ms.
            layer_ppl_impacts: Per-layer perplexity impact if converted to linear attention.
            layer_speedups: Per-layer speedup / latency savings if converted to linear attention.
            min_full_layers: Minimum number of layers that must retain full attention (default: 6).

        Returns:
            Optimal layer-type assignment vector \mathbf{z}^* \in \{0, 1\}^L.
        """
        if isinstance(cls_or_self, type):
            l_total = num_layers if num_layers is not None else 32
        else:
            l_total = cls_or_self.num_layers if num_layers is None else num_layers

        latencies = layer_latencies or {"full": 1.8, "linear": 0.5}
        t_full = latencies.get("full", 1.8)
        t_linear = latencies.get("linear", 0.5)

        # Boundary safeguards
        if l_total > 6:
            mandatory_full = set([0, 1, 2, 3, l_total - 2, l_total - 1])
        else:
            mandatory_full = set([0, max(0, l_total - 1)])

        # Perplexity impacts
        if layer_ppl_impacts is None:
            ppl_impacts = []
            for i in range(l_total):
                if i in mandatory_full:
                    ppl_impacts.append(0.08)
                else:
                    ppl_impacts.append(0.004)
        else:
            if len(layer_ppl_impacts) != l_total:
                raise ValueError(
                    f"layer_ppl_impacts length {len(layer_ppl_impacts)} does not match num_layers {l_total}"
                )
            ppl_impacts = [float(p) for p in layer_ppl_impacts]

        # Speedup/Value contributions
        if layer_speedups is not None:
            if len(layer_speedups) != l_total:
                raise ValueError(
                    f"layer_speedups length {len(layer_speedups)} does not match num_layers {l_total}"
                )
            speedups = [float(s) for s in layer_speedups]
        else:
            # Latency reduction provides linear throughput acceleration
            delta_lat = max(1e-4, t_full - t_linear)
            speedups = [delta_lat if i not in mandatory_full else 0.0 for i in range(l_total)]

        candidate_layers = [i for i in range(l_total) if i not in mandatory_full]
        max_linear_items = max(0, l_total - min_full_layers)

        # Solve exact 0-1 Knapsack ILP via Branch-and-Bound
        # Sort candidates descending by value-to-cost ratio: v_l / w_l
        sorted_candidates = sorted(
            candidate_layers,
            key=lambda idx: speedups[idx] / max(1e-9, ppl_impacts[idx]),
            reverse=True,
        )

        best_selection: List[int] = []
        best_value: float = -1.0
        best_cost: float = 0.0

        def branch_and_bound(
            index: int,
            current_cost: float,
            current_value: float,
            current_selected: List[int],
        ) -> None:
            nonlocal best_value, best_selection, best_cost
            if len(current_selected) > max_linear_items:
                return
            if current_cost > max_ppl_delta + 1e-9:
                return

            if current_value > best_value:
                best_value = current_value
                best_selection = list(current_selected)
                best_cost = current_cost

            if index >= len(sorted_candidates):
                return

            # Upper bound on remaining value
            remaining_val = sum(speedups[sorted_candidates[i]] for i in range(index, len(sorted_candidates)))
            if current_value + remaining_val <= best_value:
                return

            cand = sorted_candidates[index]
            cost = ppl_impacts[cand]
            val = speedups[cand]

            # Branch 1: Include candidate
            if current_cost + cost <= max_ppl_delta + 1e-9:
                current_selected.append(cand)
                branch_and_bound(index + 1, current_cost + cost, current_value + val, current_selected)
                current_selected.pop()

            # Branch 2: Exclude candidate
            branch_and_bound(index + 1, current_cost, current_value, current_selected)

        branch_and_bound(0, 0.0, 0.0, [])

        selected_linear = set(best_selection)
        selected_full = sorted([l for l in range(l_total) if l not in selected_linear])
        z_star = [1 if l in selected_linear else 0 for l in range(l_total)]

        baseline_latency = l_total * t_full
        optimized_latency = len(selected_full) * t_full + len(selected_linear) * t_linear
        speedup_factor = baseline_latency / max(1e-6, optimized_latency)

        if isinstance(cls_or_self, type):
            router = cls_or_self(num_layers=l_total, full_attention_layers=selected_full)
        else:
            cls_or_self.full_attention_layers = selected_full
            cls_or_self._full_set = set(selected_full)
            cls_or_self._linear_layers = sorted(list(selected_linear))
            router = cls_or_self

        metadata = {
            "num_layers": l_total,
            "full_layers_count": len(selected_full),
            "linear_layers_count": len(selected_linear),
            "accumulated_ppl_delta": float(best_cost),
            "max_ppl_delta": float(max_ppl_delta),
            "baseline_latency_ms": float(baseline_latency),
            "optimized_latency_ms": float(optimized_latency),
            "speedup_factor": float(speedup_factor),
            "pareto_optimal": True,
        }

        logger.info(
            "Phase 072: Solved binary ILP Pareto schedule for L=%d: %d linear, %d full, delta_ppl=%.4f <= %.2f",
            l_total,
            len(selected_linear),
            len(selected_full),
            best_cost,
            max_ppl_delta,
        )

        return ParetoAssignmentVector(assignment=z_star, router=router, metadata=metadata)

    # =========================================================================
    # Phase 073: Canonical Depth Schedule Generator (12, 24, 32, 48 Layers)
    # =========================================================================
    @classmethod
    def get_canonical_schedule(
        cls_or_self,
        depth: Optional[int] = None,
    ) -> CanonicalSchedule:
        """Returns precomputed canonical schedule for depth in {12, 24, 32, 48}.

        Precomputed Canonical Schedules:
            - 12 Layers: Layers 0-3 Full, Layers 4-9 Linear (6 layers), Layers 10-11 Full.
            - 24 Layers: Layers 0-3 Full, Layers 4-21 Linear (18 layers), Layers 22-23 Full.
            - 32 Layers: Layers 0-3 Full, Layers 4-29 Linear (26 layers), Layers 30-31 Full.
            - 48 Layers: Layers 0-3 Full, Layers 4-45 Linear (42 layers), Layers 46-47 Full.

        Args:
            depth: Total depth (must be positive; standard depths: 12, 24, 32, 48).

        Returns:
            CanonicalSchedule mapping layer index to 'full_attention' or 'linear_attention'.
        """
        if depth is None:
            if isinstance(cls_or_self, type):
                target_depth = 32
            else:
                target_depth = cls_or_self.num_layers
        else:
            target_depth = depth

        if target_depth <= 0:
            raise ValueError(f"depth must be positive, got {target_depth}")

        if target_depth in CANONICAL_SCHEDULE_PARTITIONS:
            spec = CANONICAL_SCHEDULE_PARTITIONS[target_depth]
            full_layers = list(spec["full_attention_layers"])
            linear_layers = list(spec["linear_attention_layers"])
        else:
            if target_depth > 6:
                full_layers = sorted(list(set([0, 1, 2, 3, target_depth - 2, target_depth - 1])))
            else:
                full_layers = sorted(list(set([0, max(0, target_depth - 1)])))
            linear_layers = [i for i in range(target_depth) if i not in set(full_layers)]

        full_set = set(full_layers)
        schedule_map = {
            i: "full_attention" if i in full_set else "linear_attention"
            for i in range(target_depth)
        }

        router_cls = cls_or_self if isinstance(cls_or_self, type) else type(cls_or_self)
        router = router_cls(num_layers=target_depth, full_attention_layers=full_layers)

        return CanonicalSchedule(
            depth=target_depth,
            full_attention_layers=full_layers,
            linear_attention_layers=linear_layers,
            schedule_map=schedule_map,
            router=router,
        )

    # =========================================================================
    # Phase 074: Routing Schedule Serialization & Round-Trip Telemetry
    # =========================================================================
    @classmethod
    def validate_schedule_schema(cls, data: Dict[str, Any]) -> None:
        """Validates routing schedule schema for 100% round-trip fidelity.

        Args:
            data: Serialized dictionary data to validate.

        Raises:
            ValueError: If schema keys, types, index ranges, or partition constraints are violated.
        """
        if not isinstance(data, dict):
            raise ValueError(f"Schema validation error: data must be a dict, got {type(data)}")

        if "num_layers" not in data:
            raise ValueError("Schema validation error: missing required key 'num_layers'")
        num_layers = data["num_layers"]
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(f"Schema validation error: 'num_layers' must be a positive int, got {num_layers}")

        if "full_attention_layers" not in data:
            raise ValueError("Schema validation error: missing required key 'full_attention_layers'")
        full_layers = data["full_attention_layers"]
        if not isinstance(full_layers, (list, tuple)):
            raise ValueError(
                f"Schema validation error: 'full_attention_layers' must be a list, got {type(full_layers)}"
            )

        if "linear_attention_layers" not in data:
            raise ValueError("Schema validation error: missing required key 'linear_attention_layers'")
        linear_layers = data["linear_attention_layers"]
        if not isinstance(linear_layers, (list, tuple)):
            raise ValueError(
                f"Schema validation error: 'linear_attention_layers' must be a list, got {type(linear_layers)}"
            )

        # Check bounds
        for idx in full_layers:
            if not isinstance(idx, int) or idx < 0 or idx >= num_layers:
                raise ValueError(
                    f"Schema validation error: full attention layer index {idx} out of range [0, {num_layers - 1}]"
                )
        for idx in linear_layers:
            if not isinstance(idx, int) or idx < 0 or idx >= num_layers:
                raise ValueError(
                    f"Schema validation error: linear attention layer index {idx} out of range [0, {num_layers - 1}]"
                )

        full_set = set(full_layers)
        linear_set = set(linear_layers)

        if full_set.intersection(linear_set):
            overlap = sorted(list(full_set.intersection(linear_set)))
            raise ValueError(
                f"Schema validation error: overlapping layers in full and linear partitions: {overlap}"
            )

        if full_set.union(linear_set) != set(range(num_layers)):
            missing = sorted(list(set(range(num_layers)) - full_set.union(linear_set)))
            raise ValueError(
                f"Schema validation error: partition incomplete, missing layers: {missing}"
            )

    def compute_memory_savings(
        self,
        uncompressed_kv_dim: int = 4096,
        latent_kv_dim: int = 512,
        seq_len: int = 2048,
        bytes_per_elem: int = 2,
    ) -> Dict[str, Any]:
        """Estimates KV cache memory savings enabled by linear attention surgery."""
        if uncompressed_kv_dim <= 0 or seq_len <= 0:
            raise ValueError("uncompressed_kv_dim and seq_len must be positive")
        num_linear = len(self._linear_layers)
        num_full = len(self.full_attention_layers)
        full_layer_kv_bytes = 2 * seq_len * uncompressed_kv_dim * bytes_per_elem
        linear_layer_state_bytes = latent_kv_dim * latent_kv_dim * bytes_per_elem

        baseline_total_bytes = self.num_layers * full_layer_kv_bytes
        optimized_total_bytes = num_full * full_layer_kv_bytes + num_linear * linear_layer_state_bytes
        savings_bytes = max(0, baseline_total_bytes - optimized_total_bytes)
        savings_ratio = savings_bytes / max(1.0, float(baseline_total_bytes))

        return {
            "uncompressed_kv_dim": uncompressed_kv_dim,
            "latent_kv_dim": latent_kv_dim,
            "seq_len": seq_len,
            "full_layers": num_full,
            "linear_layers": num_linear,
            "baseline_kv_bytes": baseline_total_bytes,
            "optimized_kv_bytes": optimized_total_bytes,
            "memory_saved_bytes": savings_bytes,
            "memory_saved_percent": float(savings_ratio * 100.0),
        }

    def compute_speedup_estimates(
        self,
        t_full: float = 1.8,
        t_linear: float = 0.5,
    ) -> Dict[str, Any]:
        """Estimates throughput speedup factor enabled by linear attention surgery."""
        if t_full <= 0 or t_linear <= 0:
            raise ValueError("Latencies t_full and t_linear must be positive")
        num_linear = len(self._linear_layers)
        num_full = len(self.full_attention_layers)
        baseline_latency = self.num_layers * t_full
        optimized_latency = num_full * t_full + num_linear * t_linear
        speedup = baseline_latency / max(1e-6, optimized_latency)
        return {
            "t_full_ms": float(t_full),
            "t_linear_ms": float(t_linear),
            "baseline_latency_ms": float(baseline_latency),
            "optimized_latency_ms": float(optimized_latency),
            "speedup_factor": float(speedup),
            "speedup_percent": float((speedup - 1.0) * 100.0),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes MoA router configuration, schedule, and telemetry to dictionary."""
        return {
            "schema_version": "1.0.0",
            "num_layers": self.num_layers,
            "full_attention_layers": list(self.full_attention_layers),
            "linear_attention_layers": list(self._linear_layers),
            "schedule": self.partition_schedule(),
            "per_layer_assignment": {str(k): v for k, v in self.partition_schedule().items()},
            "assignment_vector": [1 if i in self._linear_layers else 0 for i in range(self.num_layers)],
            "entropy": list(self._entropies),
            "memory_savings": self.compute_memory_savings(),
            "speedup_estimates": self.compute_speedup_estimates(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MoARouter:
        """Instantiates MoA router from serialized dictionary configuration with schema validation."""
        cls.validate_schedule_schema(data)
        num_layers = int(data["num_layers"])
        full_layers = [int(x) for x in data["full_attention_layers"]]
        entropies = data.get("entropy", None)
        if entropies is not None and len(entropies) == num_layers:
            entropies_list: Optional[List[float]] = [float(e) for e in entropies]
        else:
            entropies_list = None
        return cls(num_layers=num_layers, full_attention_layers=full_layers, entropies=entropies_list)

    def save_json(self, path: Union[str, Path], indent: int = 2) -> None:
        """Serializes MoA router configuration and telemetry to JSON with schema validation."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        self.validate_schedule_schema(data)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> MoARouter:
        """Loads and reconstructs a MoA router from a JSON file with schema validation."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Routing schedule file not found: {path}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        cls.validate_schedule_schema(data)
        return cls.from_dict(data)

    @classmethod
    def solve_pareto_frontier(
        cls,
        num_layers: int = 32,
        layer_latencies: Optional[Dict[str, float]] = None,
        layer_ppl_impacts: Optional[List[float]] = None,
        max_ppl_delta: float = 0.15,
        min_full_layers: int = 6,
    ) -> Tuple[MoARouter, Dict[str, Any]]:
        r"""Backwards-compatible wrapper discovering Pareto optimal partition."""
        z_vector = cls.solve_pareto_schedule(
            num_layers=num_layers,
            max_ppl_delta=max_ppl_delta,
            layer_latencies=layer_latencies,
            layer_ppl_impacts=layer_ppl_impacts,
            min_full_layers=min_full_layers,
        )
        return z_vector.router, z_vector.metadata

    @classmethod
    def compute_entropy_partition(
        cls,
        attention_matrices: Sequence[torch.Tensor],
        num_full_layers: int = 6,
        tau_entropy: Optional[float] = None,
        eps: float = 1e-9,
    ) -> MoARouter:
        r"""Dynamically computes Pareto optimal layer allocation using attention entropy profiles."""
        num_layers = len(attention_matrices)
        if num_layers == 0:
            raise ValueError("attention_matrices sequence cannot be empty")

        entropies: List[Tuple[int, float]] = []
        raw_entropies: List[float] = []
        for l_idx, attn in enumerate(attention_matrices):
            ent = compute_attention_entropy(attn, eps=eps)
            entropies.append((l_idx, ent))
            raw_entropies.append(ent)

        if tau_entropy is not None:
            full_layers = [l_idx for l_idx, ent in entropies if ent >= tau_entropy]
            if not full_layers:
                # Always retain boundary layers if none cross threshold
                full_layers = [0, max(0, num_layers - 1)]
        else:
            sorted_by_entropy = sorted(entropies, key=lambda x: x[1], reverse=True)
            k = min(num_full_layers, num_layers)
            full_layers = sorted([x[0] for x in sorted_by_entropy[:k]])

        logger.info(
            "Computed dynamic MoA entropy partition: %d full attention layers %s across %d total layers.",
            len(full_layers),
            full_layers,
            num_layers,
        )
        return cls(num_layers=num_layers, full_attention_layers=full_layers, entropies=raw_entropies)

    def summary(self) -> str:
        """Returns human-readable summary of the MoA partition schedule."""
        full_cnt = len(self.full_attention_layers)
        linear_cnt = len(self._linear_layers)
        ratio = (linear_cnt / self.num_layers) * 100.0 if self.num_layers > 0 else 0.0
        return (
            f"MoARouter(total_layers={self.num_layers}, "
            f"full_attention={full_cnt} {self.full_attention_layers}, "
            f"linear_attention={linear_cnt} [{linear_cnt} layers, {ratio:.1f}%])"
        )


# Backward-compatibility alias
HybridMoARouter = MoARouter


__all__ = [
    "compute_attention_entropy",
    "profile_layer_sensitivity",
    "AttentionEntropyProfile",
    "MiddleLayersRoutingResult",
    "ParetoAssignmentVector",
    "CanonicalSchedule",
    "MoARouter",
    "HybridMoARouter",
]
