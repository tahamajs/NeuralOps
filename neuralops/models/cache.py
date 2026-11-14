#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unified Hybrid Attention Cache

Multi-family KV and recurrent state cache managing standard attention, sliding window, and recurrent layers.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch


class UnifiedHybridCache:
    """
    Unified Hybrid Cache for heterogeneous multi-layer language models.
    Seamlessly manages per-layer state without requiring model re-configuration.
    """

    is_static: bool = True

    def __init__(self, num_layers: int | None = None) -> None:
        self.layers: list[Any] = [] if num_layers is None else [None] * num_layers
        self._seen_tokens: int = 0

    def __len__(self) -> int:
        return len(self.layers)

<<<<<<< HEAD
    def __iter__(self):
=======
    def __iter__(self) -> Any:
>>>>>>> fix/benchmarks-remediation
        return iter(self.layers)

    def __getitem__(self, idx: int) -> Any:
        if idx < len(self.layers):
            return self.layers[idx]
        return None

    def __setitem__(self, idx: int, value: Any) -> None:
        while len(self.layers) <= idx:
            self.layers.append(None)
        self.layers[idx] = value

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the current sequence length stored in the cache."""
        if self._seen_tokens > 0:
            return self._seen_tokens
        for layer_state in self.layers:
            if layer_state is None:
                continue
            if isinstance(layer_state, (tuple, list)) and len(layer_state) >= 1:
                t = layer_state[0]
                if isinstance(t, torch.Tensor):
                    if t.dim() == 4 and t.shape[-2] != t.shape[-1]:
                        return t.shape[-2]
            elif isinstance(layer_state, torch.Tensor):
                if layer_state.dim() == 4 and layer_state.shape[-2] != layer_state.shape[-1]:
                    return layer_state.shape[-2]
        return self._seen_tokens

    def get_query_offset(self, layer_idx: int = 0) -> int:
        return 0

    def get_max_cache_shape(self) -> int:
        return self._seen_tokens

    def get_usable_length(self, new_seq_len: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def update_layer(
        self,
        layer_idx: int,
        state: Any,
        layer_type: str = "linear",
        window_size: int | None = None,
        num_tokens: int | None = None,
    ) -> Any:
        """
        Updates the cache for a given layer index according to its attention mechanism type.

        Args:
            layer_idx: Target transformer layer index.
            state: Key-value tuple or recurrent state tensor for the layer.
            layer_type: Attention family identifier ('full', 'dense', 'window', 'linear', etc.).
            window_size: Sliding window retention size if applicable.
            num_tokens: Explicit token count added by this update step.

        Returns:
            Updated cache state for the target layer.
        """
        while len(self.layers) <= layer_idx:
            self.layers.append(None)

        if state is None:
            return self.layers[layer_idx]

<<<<<<< HEAD
        if isinstance(state, (tuple, list)) and len(state) >= 2:
            k = state[0]
            added = k.shape[-2] if isinstance(k, torch.Tensor) and k.ndim == 4 else 1
        elif isinstance(state, torch.Tensor):
            added = state.shape[-2] if state.ndim == 4 else 1
        else:
            added = 1

        if layer_idx == 0:
            self._seen_tokens += added
=======
        incoming_tokens = num_tokens
        if incoming_tokens is None:
            if isinstance(state, (tuple, list)) and len(state) >= 1 and isinstance(state[0], torch.Tensor):
                t = state[0]
                if t.dim() >= 2 and (t.dim() != 4 or t.shape[-2] != t.shape[-1]):
                    incoming_tokens = t.shape[-2]
            elif isinstance(state, torch.Tensor) and state.dim() >= 2:
                if state.dim() != 4 or state.shape[-2] != state.shape[-1]:
                    incoming_tokens = state.shape[-2]
>>>>>>> fix/benchmarks-remediation

        if "full" in layer_type.lower() or "dense" in layer_type.lower() or "gqa" in layer_type.lower():
            if isinstance(state, (tuple, list)) and len(state) >= 2:
                k, v = state[0], state[1]
                prev = self.layers[layer_idx]
                if prev is not None and isinstance(prev, (tuple, list)) and len(prev) >= 2:
                    k = torch.cat([prev[0], k], dim=-2)
                    v = torch.cat([prev[1], v], dim=-2)
                updated_state = (k, v)
                self.layers[layer_idx] = updated_state
                if layer_idx == 0:
                    self._seen_tokens = k.shape[-2]
                return updated_state

        if "window" in layer_type.lower() or "sliding" in layer_type.lower():
            if isinstance(state, (tuple, list)) and len(state) >= 2:
                k, v = state[0], state[1]
                prev = self.layers[layer_idx]
                if prev is not None and isinstance(prev, (tuple, list)) and len(prev) >= 2:
                    k = torch.cat([prev[0], k], dim=-2)
                    v = torch.cat([prev[1], v], dim=-2)
                if window_size is not None and k.shape[-2] > window_size:
                    k = k[..., -window_size:, :]
                    v = v[..., -window_size:, :]
                updated_state = (k, v)
                self.layers[layer_idx] = updated_state
                if layer_idx == 0:
                    added = incoming_tokens if incoming_tokens is not None else 1
                    self._seen_tokens += added
                return updated_state

        self.layers[layer_idx] = state
        if layer_idx == 0 and incoming_tokens is not None:
            self._seen_tokens += incoming_tokens
        return state

<<<<<<< HEAD
    def reorder_cache(self, beam_idx: torch.LongTensor) -> UnifiedHybridCache:
        """Reorders cache states according to beam search indices."""
        reordered = UnifiedHybridCache(len(self.layers))
        reordered._seen_tokens = self._seen_tokens
        for i, layer_state in enumerate(self.layers):
            if layer_state is None:
                reordered.layers[i] = None
            elif isinstance(layer_state, tuple):
                reordered.layers[i] = tuple(
                    s.index_select(0, beam_idx.to(s.device)) if isinstance(s, torch.Tensor) else s
                    for s in layer_state
                )
            elif isinstance(layer_state, list):
                reordered.layers[i] = [
                    s.index_select(0, beam_idx.to(s.device)) if isinstance(s, torch.Tensor) else s
                    for s in layer_state
                ]
            elif isinstance(layer_state, torch.Tensor):
                reordered.layers[i] = layer_state.index_select(0, beam_idx.to(layer_state.device))
            else:
                reordered.layers[i] = layer_state
=======
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """HF Cache protocol update method."""
        layer_type = "full"
        window_size = None
        if cache_kwargs:
            layer_type = cache_kwargs.get("layer_type", "full")
            window_size = cache_kwargs.get("window_size", None)
        num_toks = key_states.shape[-2] if key_states.dim() >= 2 else 1
        return self.update_layer(
            layer_idx=layer_idx,
            state=(key_states, value_states),
            layer_type=layer_type,
            window_size=window_size,
            num_tokens=num_toks,
        )

    def reorder_cache(self, beam_idx: torch.Tensor) -> UnifiedHybridCache:
        """
        Reorders cache states according to beam_idx along the batch dimension.

        Args:
            beam_idx: Tensor containing indices of selected beams.

        Returns:
            New UnifiedHybridCache with reordered states.
        """
        reordered = UnifiedHybridCache(len(self.layers))
        reordered._seen_tokens = self._seen_tokens
        for idx, layer_state in enumerate(self.layers):
            if layer_state is None:
                reordered[idx] = None
            elif isinstance(layer_state, (tuple, list)):
                reordered[idx] = tuple(
                    s.index_select(0, beam_idx.to(s.device)) if isinstance(s, torch.Tensor) else s
                    for s in layer_state
                )
            elif isinstance(layer_state, torch.Tensor):
                reordered[idx] = layer_state.index_select(0, beam_idx.to(layer_state.device))
            else:
                reordered[idx] = layer_state
>>>>>>> fix/benchmarks-remediation
        return reordered

    def update_seen_tokens(self, num_tokens: int) -> None:
        self._seen_tokens += num_tokens

    def to_legacy_cache(self) -> tuple[Any, ...]:
        return tuple(self.layers)

    @classmethod
    def from_legacy_cache(cls, past_key_values: tuple[Any, ...] | list[Any]) -> UnifiedHybridCache:
        cache = cls(len(past_key_values))
        for i, val in enumerate(past_key_values):
            cache[i] = val
        return cache


__all__ = ["UnifiedHybridCache"]
