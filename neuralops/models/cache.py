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

from typing import Any, Tuple, List, Optional
import torch


class UnifiedHybridCache:
    """
    Unified Hybrid Cache for heterogeneous multi-layer language models.
    Seamlessly manages per-layer state without requiring model re-configuration.
    """

    def __init__(self, num_layers: int | None = None) -> None:
        self.layers: list[Any] = [] if num_layers is None else [None] * num_layers
        self._seen_tokens: int = 0

    def __len__(self) -> int:
        return len(self.layers)

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
    ) -> Any:
        """
        Updates the cache for a given layer index according to its attention mechanism type.
        """
        while len(self.layers) <= layer_idx:
            self.layers.append(None)

        if state is None:
            return self.layers[layer_idx]

        if "full" in layer_type.lower() or "dense" in layer_type.lower() or "gqa" in layer_type.lower():
            if isinstance(state, (tuple, list)) and len(state) >= 2:
                k, v = state[0], state[1]
                prev = self.layers[layer_idx]
                if prev is not None and isinstance(prev, (tuple, list)) and len(prev) >= 2:
                    k = torch.cat([prev[0], k], dim=-2)
                    v = torch.cat([prev[1], v], dim=-2)
                updated_state = (k, v)
                self.layers[layer_idx] = updated_state
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
                return updated_state

        self.layers[layer_idx] = state
        return state

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
