#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
HuggingFace DynamicCache Generation Bridge

Bridges NeuralOps recurrent and hybrid state caches with HuggingFace DynamicCache.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import torch

logger = logging.getLogger("NeuralOps.Surgery.HFCacheBridge")

from neuralops.core.errors import IntegrationNotImplementedError

try:
    from transformers.cache_utils import Cache, DynamicCache
    _HF_CACHE_BASE = DynamicCache
except (ImportError, AttributeError):
    class Cache:
        pass
    class DynamicCache:
        pass
    _HF_CACHE_BASE = object


def validate_cache_format(cache: Any) -> None:
    """Validate that the given cache format is supported by NeuralOps, failing closed otherwise."""
    if cache is None:
        return
    cache_type_name = type(cache).__name__
    unsupported = ("StaticCache", "OffloadedCache", "QuantizedCache", "EncoderDecoderCache", "SinkCache")
    if any(u in cache_type_name for u in unsupported) or (
        isinstance(cache, Cache) and not isinstance(cache, (DynamicCache, NeuralOpsHybridHFCache))
    ):
        raise IntegrationNotImplementedError(
            f"Unsupported HuggingFace cache format: '{cache_type_name}'. "
            "NeuralOps HFCacheBridge supports DynamicCache, NeuralOpsHybridHFCache, and standard tuple/list caches."
        )


class NeuralOpsHybridHFCache(_HF_CACHE_BASE):
    """
    HuggingFace compliant generation cache supporting heterogeneous layers.
    Guarantees get_seq_length(), get_max_length(), and get_usable_length()
    track true sequence lengths so RoPE position_ids and cache_position match exactly.
    Properly handles both standard 4D KV tensors (B, H, L, D) and 4D linear recurrent
    state matrices (B, H, Dk, Dv) without crashing or slicing incompatibilities during generation.
    """

    def __init__(self) -> None:
        if _HF_CACHE_BASE is not object and hasattr(super(), "__init__"):
            super().__init__()
        # Transformers 4.x exposes these lists directly, while newer releases
        # store per-layer objects in ``layers``. Keep a compatibility view for
        # tuple-style callers without replacing the base cache's own lists.
        if not hasattr(self, "key_cache"):
            self.key_cache: List[torch.Tensor] = []
        if not hasattr(self, "value_cache"):
            self.value_cache: List[torch.Tensor] = []
        self.recurrent_state_cache: Dict[int, Any] = {}
        self.sliding_window_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._recurrent_layers: set[int] = set()
        self._seen_tokens: int = 0

    def is_recurrent_layer(self, layer_idx: int) -> bool:
        """Return True if the specified layer index holds recurrent state."""
        return layer_idx in self.recurrent_state_cache or layer_idx in self._recurrent_layers

    def mark_recurrent_layer(self, layer_idx: int) -> None:
        """Explicitly register a layer index as using linear recurrent state semantics."""
        self._recurrent_layers.add(layer_idx)

    def _is_recurrent_tensor(self, tensor: torch.Tensor, layer_idx: Optional[int] = None) -> bool:
        """Determine whether a tensor is a 4D recurrent state matrix (B, H, Dk, Dv)."""
        if layer_idx is not None and self.is_recurrent_layer(layer_idx):
            return True
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
            return False
        # Recurrent state matrices (B, H, Dk, Dv) have head dimensions (typically Dk == Dv >= 8)
        # and do not change sequence length during decoding.
        if tensor.shape[-2] == tensor.shape[-1] and tensor.shape[-2] >= 8:
            if self._seen_tokens > 0 and tensor.shape[-2] != self._seen_tokens:
                return True
        return False

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Return a layer's cached length, or the processed-token count for recurrent layers."""
        if layer_idx is not None:
            if self.is_recurrent_layer(layer_idx):
                return self._seen_tokens
            if 0 <= layer_idx < len(self.key_cache):
                key_states = self.key_cache[layer_idx]
                if key_states.numel() > 0:
                    if self._is_recurrent_tensor(key_states, layer_idx):
                        return self._seen_tokens
                    return int(key_states.shape[-2])
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        return None

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Return the unbounded capacity used by Transformers cache APIs."""
        return -1

    def get_usable_length(self, new_seq_len: int, layer_idx: Optional[int] = 0) -> int:
        return self.get_seq_length(layer_idx)

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> Tuple[int, int]:
        """Return full-attention mask length and zero offset for dynamic KV layers."""
        return self.get_seq_length(layer_idx) + query_length, 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: Optional[torch.Tensor] = None,
        layer_idx: int = 0,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates cache for full-attention KV layers or constant O(1) linear recurrent layers.
        Properly distinguishes 4D KV tensors (B, H, L, D) from 4D recurrent state matrices (B, H, Dk, Dv).
        """
        seen_before_update = self._seen_tokens

        # Check if this update represents a recurrent state matrix
        is_recurrent = (
            self.is_recurrent_layer(layer_idx)
            or (cache_kwargs is not None and (
                cache_kwargs.get("is_recurrent", False)
                or cache_kwargs.get("recurrent", False)
                or cache_kwargs.get("layer_type") in ("recurrent", "linear", "linear_attn", "gla", "retnet", "rwkv")
            ))
            or value_states is None
            or (value_states is key_states and key_states.ndim == 4)
        )

        if not is_recurrent and key_states.ndim == 4:
            if layer_idx < len(self.key_cache) and self.key_cache[layer_idx].numel() > 0:
                cached = self.key_cache[layer_idx]
                if cached.ndim == 4 and cached.shape == key_states.shape and key_states.shape[-2] > 1:
                    is_recurrent = True
            elif key_states.shape[-2] == key_states.shape[-1] and key_states.shape[-2] >= 8 and seen_before_update > 0 and key_states.shape[-2] != 1:
                is_recurrent = True

        effective_val = value_states if value_states is not None else key_states

        if is_recurrent:
            self._recurrent_layers.add(layer_idx)
            self.recurrent_state_cache[layer_idx] = key_states

            while len(self.key_cache) <= layer_idx:
                self.key_cache.append(torch.empty(0, device=key_states.device))
                self.value_cache.append(torch.empty(0, device=effective_val.device))

            # Store recurrent state matrix without concatenating along dim=-2
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = effective_val

            num_tokens = 1
            if cache_kwargs is not None:
                if "num_tokens" in cache_kwargs:
                    num_tokens = int(cache_kwargs["num_tokens"])
                elif "cache_position" in cache_kwargs:
                    cp = cache_kwargs["cache_position"]
                    if isinstance(cp, torch.Tensor):
                        num_tokens = int(cp.numel())
            elif seen_before_update == 0:
                num_tokens = 1

            if layer_idx == 0:
                self._seen_tokens = seen_before_update + num_tokens

            self._sync_base_layers()
            return key_states, effective_val

        # Standard KV Cache (B, H, L, D)
        if _HF_CACHE_BASE is not object:
            updated_key, updated_value = super().update(
                key_states,
                effective_val,
                layer_idx,
                cache_kwargs=cache_kwargs,
            )
        else:
            # The fallback cache is used when Transformers is not installed.
            while len(self.key_cache) <= layer_idx:
                self.key_cache.append(torch.empty(0, device=key_states.device))
                self.value_cache.append(torch.empty(0, device=effective_val.device))
            if self.key_cache[layer_idx].numel() == 0:
                updated_key, updated_value = key_states, effective_val
            else:
                updated_key = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                updated_value = torch.cat([self.value_cache[layer_idx], effective_val], dim=-2)
            self.key_cache[layer_idx] = updated_key
            self.value_cache[layer_idx] = updated_value

        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(torch.empty(0, device=key_states.device))
            self.value_cache.append(torch.empty(0, device=effective_val.device))
        self.key_cache[layer_idx] = updated_key
        self.value_cache[layer_idx] = updated_value

        if layer_idx == 0:
            self._seen_tokens = seen_before_update + key_states.shape[-2]

        self._sync_base_layers()
        return updated_key, updated_value

    def update_recurrent_state(
        self,
        arg1: Any,
        arg2: Any = None,
        num_tokens: int = 1,
        **kwargs: Any,
    ) -> Any:
        """
        Stores constant O(1) recurrent states (GLA, GDN, RetNet, etc.).
        Supports both (layer_idx, state) and HuggingFace Cache (recurrent_states, layer_idx) signatures.
        """
        if isinstance(arg1, int):
            layer_idx = arg1
            state = arg2
        elif isinstance(arg2, int):
            state = arg1
            layer_idx = arg2
        else:
            raise ValueError(
                f"One of the first two arguments must be an integer layer_idx, got {type(arg1)} and {type(arg2)}"
            )

        num_tokens = kwargs.get("num_tokens", num_tokens)
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")

        self._recurrent_layers.add(layer_idx)
        self.recurrent_state_cache[layer_idx] = state

        if isinstance(state, torch.Tensor):
            while len(self.key_cache) <= layer_idx:
                self.key_cache.append(torch.empty(0, device=state.device))
                self.value_cache.append(torch.empty(0, device=state.device))
            self.key_cache[layer_idx] = state
            self.value_cache[layer_idx] = state

        if layer_idx == 0:
            self._seen_tokens += num_tokens

        self._sync_base_layers()
        return state

    def update_sliding_window_cache(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        window_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append KV states and retain only the newest ``window_size`` tokens."""
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if key_states.shape != value_states.shape or key_states.ndim < 3:
            raise ValueError("Sliding-window key/value states must have matching rank-3+ shapes")
        previous = self.sliding_window_cache.get(layer_idx)
        if previous is None:
            full_key, full_value = key_states, value_states
        else:
            old_key, old_value = previous
            full_key = torch.cat((old_key, key_states), dim=-2)
            full_value = torch.cat((old_value, value_states), dim=-2)
        cached = (full_key[..., -window_size:, :], full_value[..., -window_size:, :])
        self.sliding_window_cache[layer_idx] = cached
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return cached

    def get_recurrent_state(self, layer_idx: int) -> Optional[Any]:
        if layer_idx in self.recurrent_state_cache:
            return self.recurrent_state_cache[layer_idx]
        if layer_idx < len(self.key_cache) and self.is_recurrent_layer(layer_idx):
            return self.key_cache[layer_idx]
        return None

    @staticmethod
    def _map_batch_state(state: Any, transform: Any) -> Any:
        if isinstance(state, torch.Tensor) and state.ndim > 0:
            return transform(state)
        if isinstance(state, tuple):
            return tuple(NeuralOpsHybridHFCache._map_batch_state(item, transform) for item in state)
        if isinstance(state, list):
            return [NeuralOpsHybridHFCache._map_batch_state(item, transform) for item in state]
        if isinstance(state, dict):
            return {key: NeuralOpsHybridHFCache._map_batch_state(value, transform) for key, value in state.items()}
        return state

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder KV and recurrent states together for beam-search selection."""
        for index, (key, value) in enumerate(zip(self.key_cache, self.value_cache)):
            if key.numel() == 0:
                continue
            self.key_cache[index] = key.index_select(0, beam_idx.to(key.device))
            self.value_cache[index] = value.index_select(0, beam_idx.to(value.device))
        self.recurrent_state_cache = {
            index: self._map_batch_state(
                state, lambda tensor: tensor.index_select(0, beam_idx.to(tensor.device))
            )
            for index, state in self.recurrent_state_cache.items()
        }
        self.sliding_window_cache = {
            index: (
                key.index_select(0, beam_idx.to(key.device)),
                value.index_select(0, beam_idx.to(value.device)),
            )
            for index, (key, value) in self.sliding_window_cache.items()
        }
        self._sync_base_layers()

    def batch_repeat_interleave(self, repeats: int) -> None:
        """Repeat every batch-indexed KV and recurrent state for beam expansion."""
        if repeats < 1:
            raise ValueError("repeats must be at least one")
        for index, (key, value) in enumerate(zip(self.key_cache, self.value_cache)):
            if key.numel() == 0:
                continue
            self.key_cache[index] = key.repeat_interleave(repeats, dim=0)
            self.value_cache[index] = value.repeat_interleave(repeats, dim=0)
        self.recurrent_state_cache = {
            index: self._map_batch_state(state, lambda tensor: tensor.repeat_interleave(repeats, dim=0))
            for index, state in self.recurrent_state_cache.items()
        }
        self.sliding_window_cache = {
            index: (
                key.repeat_interleave(repeats, dim=0),
                value.repeat_interleave(repeats, dim=0),
            )
            for index, (key, value) in self.sliding_window_cache.items()
        }
        self._sync_base_layers()

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        """Select batch rows consistently across KV tensors and recurrent states."""
        for index, (key, value) in enumerate(zip(self.key_cache, self.value_cache)):
            if key.numel() == 0:
                continue
            self.key_cache[index] = key.index_select(0, indices.to(key.device))
            self.value_cache[index] = value.index_select(0, indices.to(value.device))
        self.recurrent_state_cache = {
            index: self._map_batch_state(
                state, lambda tensor: tensor.index_select(0, indices.to(tensor.device))
            )
            for index, state in self.recurrent_state_cache.items()
        }
        self.sliding_window_cache = {
            index: (
                key.index_select(0, indices.to(key.device)),
                value.index_select(0, indices.to(value.device)),
            )
            for index, (key, value) in self.sliding_window_cache.items()
        }
        self._sync_base_layers()

    def crop(self, max_length: int) -> None:
        """Crop full-attention caches; preserves recurrent states without slicing."""
        target_length = max_length if max_length >= 0 else max(0, self._seen_tokens + max_length)
        if self._seen_tokens <= target_length and max_length >= 0:
            return

        for index in range(len(self.key_cache)):
            key = self.key_cache[index]
            value = self.value_cache[index] if index < len(self.value_cache) else None
            if key.numel() == 0:
                continue
            # For recurrent states (B, H, Dk, Dv), do not slice feature dimensions
            if self.is_recurrent_layer(index) or self._is_recurrent_tensor(key, index):
                continue
            self.key_cache[index] = key[..., :target_length, :]
            if value is not None and value.numel() > 0:
                self.value_cache[index] = value[..., :target_length, :]

        for index, (w_key, w_val) in list(self.sliding_window_cache.items()):
            if w_key.shape[-2] > target_length:
                self.sliding_window_cache[index] = (
                    w_key[..., :target_length, :],
                    w_val[..., :target_length, :],
                )

        self._seen_tokens = min(self._seen_tokens, target_length)
        self._sync_base_layers()

    def reset(self) -> None:
        """Clear all attention and recurrent states for a new generation request."""
        self.key_cache.clear()
        self.value_cache.clear()
        self.recurrent_state_cache.clear()
        self.sliding_window_cache.clear()
        self._recurrent_layers.clear()
        self._seen_tokens = 0
        if hasattr(self, "layers"):
            self.layers.clear()

    def _sync_base_layers(self) -> None:
        """Keep Transformers' layer-backed cache view in sync when present."""
        for index, layer in enumerate(getattr(self, "layers", [])):
            if index >= len(self.key_cache) or self.key_cache[index].numel() == 0:
                continue
            if hasattr(layer, "keys"):
                layer.keys = self.key_cache[index]
                layer.values = self.value_cache[index]
                layer.is_initialized = True
                if self.is_recurrent_layer(index):
                    layer.get_seq_length = lambda: self._seen_tokens
                    layer.crop = lambda max_length: None

    def __getitem__(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Index access returning (key, value) tuple for layer_idx, compatible with legacy tuple caches."""
        if layer_idx < 0:
            layer_idx += len(self)
        if 0 <= layer_idx < len(self.key_cache):
            key_states = self.key_cache[layer_idx]
            value_states = self.value_cache[layer_idx]
            if key_states.numel() > 0:
                return key_states, value_states
        if layer_idx in self.recurrent_state_cache:
            state = self.recurrent_state_cache[layer_idx]
            if isinstance(state, tuple) and len(state) == 2:
                return state
            return state, state
        if layer_idx in self.sliding_window_cache:
            return self.sliding_window_cache[layer_idx]
        if layer_idx >= len(self):
            raise IndexError(f"Layer index {layer_idx} out of range for cache with length {len(self)}")
        return None, None

    def __len__(self) -> int:
        return max(
            len(self.key_cache),
            len(self.recurrent_state_cache),
            len(self.sliding_window_cache),
        )

    def __iter__(self) -> Iterator[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """Iterate by layer using the legacy ``(key, value)`` view."""
        for layer_idx in range(len(self)):
            yield self[layer_idx]

    def to_legacy_cache(
        self,
    ) -> Tuple[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]], ...]:
        """Return the conventional tuple cache for legacy model implementations."""
        return tuple(
            self[index]
            for index in range(len(self))
        )

    @staticmethod
    def convert_recurrent_state(
        state: torch.Tensor,
        target_format: str = "recurrent",
    ) -> torch.Tensor:
        """Convert between (B, H, L, D) token sequences and (B, H, Dk, Dv) recurrent states.

        Args:
            state: 4D Tensor containing either token sequence or recurrent state matrix.
            target_format: 'recurrent' for (B, H, Dk, Dv) or 'kv' / 'sequence' for (B, H, L, D).

        Returns:
            Transformed Tensor matching the target representation contract.
        """
        if not isinstance(state, torch.Tensor) or state.ndim != 4:
            return state

        if target_format == "recurrent":
            # If in sequence format (B, H, L, D) with L != D, compute outer product state (B, H, D, D)
            if state.shape[-2] != state.shape[-1]:
                k = state.float()
                return torch.matmul(k.transpose(-2, -1), k).to(state.dtype)
            return state
        elif target_format in ("kv", "sequence"):
            # If in recurrent matrix format (B, H, Dk, Dv), project to sequence summary (B, H, 1, Dv)
            if state.shape[-2] == state.shape[-1]:
                return state.mean(dim=-2, keepdim=True)
            return state
        return state

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Any] = None,
    ) -> NeuralOpsHybridHFCache:
        """Construct a compatible cache from a legacy tuple of key/value tensors, DynamicCache, or recurrent states."""
        validate_cache_format(past_key_values)
        if isinstance(past_key_values, cls):
            return past_key_values

        cache = cls()
        if past_key_values is not None:
            # Handle native transformers.DynamicCache objects gracefully
            if isinstance(past_key_values, DynamicCache) or type(past_key_values).__name__ == "DynamicCache":
                if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
                    for layer_idx in range(len(past_key_values.key_cache)):
                        k = past_key_values.key_cache[layer_idx]
                        v = past_key_values.value_cache[layer_idx]
                        if k is not None and v is not None:
                            cache.update(k, v, layer_idx)
                elif hasattr(past_key_values, "layers"):
                    for layer_idx, layer in enumerate(past_key_values.layers):
                        k = getattr(layer, "keys", getattr(layer, "key", None))
                        v = getattr(layer, "values", getattr(layer, "value", None))
                        if k is not None and v is not None:
                            cache.update(k, v, layer_idx)
                else:
                    try:
                        for layer_idx, item in enumerate(past_key_values):
                            if item is None:
                                continue
                            if isinstance(item, (tuple, list)) and len(item) >= 2:
                                cache.update(item[0], item[1], layer_idx)
                    except Exception as err:
                        raise IntegrationNotImplementedError(
                            f"DynamicCache extraction failed for type '{type(past_key_values).__name__}': {err}"
                        ) from err
                return cache

            if not isinstance(past_key_values, (tuple, list)):
                raise IntegrationNotImplementedError(
                    f"Unsupported past_key_values format: '{type(past_key_values).__name__}'. "
                    "Expected tuple, list, or transformers.DynamicCache."
                )

            for layer_idx, item in enumerate(past_key_values):
                if item is None:
                    continue
                if isinstance(item, (tuple, list)):
                    if len(item) >= 2:
                        k, v = item[0], item[1]
                        if k is not None and v is not None:
                            cache.update(k, v, layer_idx)
                    elif len(item) == 1 and item[0] is not None:
                        cache.update_recurrent_state(layer_idx, item[0])
                elif isinstance(item, torch.Tensor):
                    cache.update_recurrent_state(layer_idx, item)
        return cache


def _compute_prefix_hash(tokens: Sequence[int] | torch.Tensor) -> str:
    """
    Computes a deterministic SHA-256 hash for a token sequence prefix (Phase 091).
    Guarantees collision-free matching for shared prompt prefix caching in SGLang radix trees.
    """
    if isinstance(tokens, torch.Tensor):
        if tokens.ndim > 1:
            tokens = tokens.squeeze()
        token_tuple = tuple(int(t) for t in tokens.tolist())
    else:
        token_tuple = tuple(int(t) for t in tokens)

    hasher = hashlib.sha256()
    for tid in token_tuple:
        hasher.update(int(tid).to_bytes(4, byteorder="big", signed=True))
    return hasher.hexdigest()


class RadixTreeNode:
    """A node in the SGLang Radix Tree storing recurrent associative state (Phase 090)."""

    def __init__(
        self,
        token_chunk: Tuple[int, ...] = (),
        parent: Optional[RadixTreeNode] = None,
        prefix_state: Optional[Dict[int, torch.Tensor]] = None,
    ) -> None:
        self.token_chunk = token_chunk
        self.parent = parent
        self.prefix_state: Optional[Dict[int, torch.Tensor]] = prefix_state
        self.children: Dict[int, RadixTreeNode] = {}
        self.access_count: int = 0
        self.last_accessed: float = time.time()

    @property
    def token_ids(self) -> Tuple[int, ...]:
        """Alias for token_chunk adhering to Phase 090 specification."""
        return self.token_chunk

    @token_ids.setter
    def token_ids(self, val: Sequence[int]) -> None:
        self.token_chunk = tuple(int(v) for v in val)

    @property
    def prefix_hash(self) -> str:
        """Deterministic SHA-256 prefix hash of this node's token chunk."""
        return _compute_prefix_hash(self.token_chunk)

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class SGLangRadixStateAdapter:
    """
    SGLang Radix Tree Recurrent State Adapter.

    Caches associative state matrices S_prefix at radix tree prefix nodes to allow O(1) state recovery
    and zero-prefill multi-turn serving for linear attention models.
    """

    def __init__(self, max_cached_nodes: int = 1000) -> None:
        self.root = RadixTreeNode()
        self.max_cached_nodes = max_cached_nodes
        self._node_count: int = 1

    @staticmethod
    def _normalize_tokens(tokens: Sequence[int] | torch.Tensor) -> Tuple[int, ...]:
        if isinstance(tokens, torch.Tensor):
            if tokens.ndim > 1:
                tokens = tokens.squeeze()
            return tuple(int(t) for t in tokens.tolist())
        return tuple(int(t) for t in tokens)

    @staticmethod
    def _extract_state_dict(
        state: Union[Dict[int, torch.Tensor], NeuralOpsHybridHFCache, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        if isinstance(state, NeuralOpsHybridHFCache):
            state_dict = {}
            for l_idx, s in state.recurrent_state_cache.items():
                if isinstance(s, torch.Tensor):
                    state_dict[l_idx] = s.detach().clone()
            for l_idx, (k, _) in enumerate(state.key_cache):
                if l_idx not in state_dict and isinstance(k, torch.Tensor) and k.numel() > 0:
                    state_dict[l_idx] = k.detach().clone()
            return state_dict
        elif isinstance(state, dict):
            return {k: v.detach().clone() for k, v in state.items() if isinstance(v, torch.Tensor)}
        elif isinstance(state, torch.Tensor):
            return {0: state.detach().clone()}
        else:
            raise TypeError(f"Unsupported state format: {type(state)}")

    def insert(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        state: Union[Dict[int, torch.Tensor], NeuralOpsHybridHFCache, torch.Tensor],
    ) -> RadixTreeNode:
        """
        Inserts a token sequence and its corresponding recurrent state S_prefix into the Radix Tree.
        Handles radix node splitting when a common prefix diverges.
        """
        tokens = self._normalize_tokens(token_ids)
        if len(tokens) == 0:
            return self.root

        state_dict = self._extract_state_dict(state)
        curr = self.root
        idx = 0

        while idx < len(tokens):
            first_tok = tokens[idx]
            if first_tok not in curr.children:
                # No child with this token, create new child with remaining tokens
                new_node = RadixTreeNode(
                    token_chunk=tokens[idx:],
                    parent=curr,
                    prefix_state=state_dict,
                )
                curr.children[first_tok] = new_node
                self._node_count += 1
                self._check_capacity()
                return new_node

            child = curr.children[first_tok]
            chunk = child.token_chunk
            # Find common prefix between chunk and tokens[idx:]
            match_len = 0
            while (
                match_len < len(chunk)
                and idx + match_len < len(tokens)
                and chunk[match_len] == tokens[idx + match_len]
            ):
                match_len += 1

            if match_len == len(chunk):
                # Full match of child's chunk, continue deeper
                idx += match_len
                curr = child
                if idx == len(tokens):
                    # Exact match at this node, update state
                    curr.prefix_state = state_dict
                    curr.last_accessed = time.time()
                    curr.access_count += 1
                    return curr
            else:
                # Partial match: split child into parent (split_node) and remaining children
                common_chunk = chunk[:match_len]
                remaining_child_chunk = chunk[match_len:]

                split_node = RadixTreeNode(
                    token_chunk=common_chunk,
                    parent=curr,
                    prefix_state=None,
                )
                curr.children[first_tok] = split_node

                # Update existing child
                child.token_chunk = remaining_child_chunk
                child.parent = split_node
                split_node.children[remaining_child_chunk[0]] = child

                self._node_count += 1
                idx += match_len

                if idx == len(tokens):
                    # The inserted sequence ends at the split node
                    split_node.prefix_state = state_dict
                    split_node.last_accessed = time.time()
                    split_node.access_count += 1
                    self._check_capacity()
                    return split_node
                else:
                    # Create second child from split node
                    new_child_chunk = tokens[idx:]
                    new_node = RadixTreeNode(
                        token_chunk=new_child_chunk,
                        parent=split_node,
                        prefix_state=state_dict,
                    )
                    split_node.children[new_child_chunk[0]] = new_node
                    self._node_count += 1
                    self._check_capacity()
                    return new_node

        curr.prefix_state = state_dict
        curr.last_accessed = time.time()
        curr.access_count += 1
        return curr

    def match_prefix(
        self,
        token_ids: Sequence[int] | torch.Tensor,
    ) -> Tuple[int, Optional[Dict[int, torch.Tensor]], Optional[RadixTreeNode]]:
        """
        Finds the longest matching token prefix in the Radix Tree.

        Returns:
            (matched_length, prefix_state_dict, matching_node)
        """
        tokens = self._normalize_tokens(token_ids)
        if len(tokens) == 0:
            return 0, None, None

        curr = self.root
        idx = 0
        deepest_node_with_state: Optional[RadixTreeNode] = None
        matched_state_len = 0

        while idx < len(tokens):
            first_tok = tokens[idx]
            if first_tok not in curr.children:
                break

            child = curr.children[first_tok]
            chunk = child.token_chunk
            match_len = 0
            while (
                match_len < len(chunk)
                and idx + match_len < len(tokens)
                and chunk[match_len] == tokens[idx + match_len]
            ):
                match_len += 1

            if match_len == len(chunk):
                idx += match_len
                curr = child
                curr.last_accessed = time.time()
                curr.access_count += 1
                if curr.prefix_state is not None:
                    deepest_node_with_state = curr
                    matched_state_len = idx
            else:
                # Partial match on edge does not yield state at intermediate chunk
                break

        if deepest_node_with_state is not None and deepest_node_with_state.prefix_state is not None:
            # Clone states for return
            res_state = {
                k: v.clone() for k, v in deepest_node_with_state.prefix_state.items()
            }
            return matched_state_len, res_state, deepest_node_with_state

        return 0, None, None

    @staticmethod
    def _compute_prefix_hash(tokens: Sequence[int] | torch.Tensor) -> str:
        """Deterministic prefix hash generator for shared system prompts (Phase 091)."""
        return _compute_prefix_hash(tokens)

    def compute_prefix_hash(self, tokens: Sequence[int] | torch.Tensor) -> str:
        """Deterministic prefix hash generator for shared system prompts (Phase 091)."""
        return _compute_prefix_hash(tokens)

    def lookup_prefix_state(
        self,
        token_ids: Sequence[int] | torch.Tensor,
    ) -> Optional[Dict[int, torch.Tensor]]:
        """
        Looks up associative state matrix S_prefix in O(1) time (Phase 092).
        If matching prefix with cached state is found, returns the recurrent state dict
        to bypass prompt prefill computation.
        """
        _, state_dict, _ = self.match_prefix(token_ids)
        return state_dict

    def cache_prefix_state(
        self,
        prompt_tokens: Sequence[int] | torch.Tensor,
        cache: Union[Dict[int, torch.Tensor], NeuralOpsHybridHFCache, torch.Tensor],
    ) -> RadixTreeNode:
        """Stores associative state matrix S_prefix at radix tree prefix node."""
        return self.insert(prompt_tokens, cache)

    def retrieve_prefix_state(
        self,
        prompt_tokens: Sequence[int] | torch.Tensor,
    ) -> Tuple[int, Optional[NeuralOpsHybridHFCache]]:
        """
        Retrieves matching prefix state in O(1) time and initializes a NeuralOpsHybridHFCache.

        Returns:
            (matched_length, populated_hybrid_cache_or_none)
        """
        matched_len, state_dict, _ = self.match_prefix(prompt_tokens)
        if state_dict is None or matched_len == 0:
            return 0, None

        cache = NeuralOpsHybridHFCache()
        cache._seen_tokens = matched_len
        for layer_idx, s in state_dict.items():
            cache.update_recurrent_state(layer_idx, s, num_tokens=0)

        return matched_len, cache

    def _check_capacity(self) -> None:
        if self._node_count > self.max_cached_nodes:
            self.evict_lru(max_nodes=self.max_cached_nodes)

    def evict_lru(
        self,
        max_nodes: Optional[int] = None,
        num_nodes: Optional[int] = None,
    ) -> int:
        """
        Evicts the least-recently used leaf nodes from the Radix Tree (Phase 094).
        If max_nodes is specified, prunes until total node count <= max_nodes.
        If num_nodes is specified, prunes up to num_nodes leaf nodes.
        """
        if max_nodes is not None:
            target_evict = max(0, self._node_count - max_nodes)
        elif num_nodes is not None:
            target_evict = num_nodes
        else:
            target_evict = 1

        if target_evict <= 0:
            return 0

        evicted = 0
        while evicted < target_evict:
            leaves: List[RadixTreeNode] = []

            def collect_leaves(node: RadixTreeNode) -> None:
                if node is not self.root and node.is_leaf():
                    leaves.append(node)
                for child in node.children.values():
                    collect_leaves(child)

            collect_leaves(self.root)
            if not leaves:
                break

            leaves.sort(key=lambda n: n.last_accessed)
            batch_to_remove = leaves[: target_evict - evicted]
            for node in batch_to_remove:
                if node.parent is not None and len(node.token_chunk) > 0:
                    first_tok = node.token_chunk[0]
                    if first_tok in node.parent.children:
                        del node.parent.children[first_tok]
                        self._node_count -= 1
                        evicted += 1
                        if evicted >= target_evict:
                            break

        return evicted

    def total_nodes(self) -> int:
        return self._node_count

    def clear(self) -> None:
        self.root = RadixTreeNode()
        self._node_count = 1


def wrap_model_for_generation(model: torch.nn.Module) -> torch.nn.Module:
    """
    Wraps model's prepare_inputs_for_generation or injects NeuralOpsHybridHFCache
    whenever model.generate() is called.
    """
    if (
        hasattr(model, "generate")
        and hasattr(model, "prepare_inputs_for_generation")
        and not getattr(model, "_neuralops_hf_cache_bridge", False)
    ):
        orig_prep = model.prepare_inputs_for_generation

        def _patched_prep(input_ids: torch.Tensor, **kwargs: Any) -> Dict[str, Any]:
            model_inputs = orig_prep(input_ids, **kwargs)
            if model_inputs.get("past_key_values", None) is None and kwargs.get("use_cache", True):
                past = kwargs.get("past_key_values")
                if isinstance(past, NeuralOpsHybridHFCache):
                    model_inputs["past_key_values"] = past
                elif past is not None:
                    model_inputs["past_key_values"] = NeuralOpsHybridHFCache.from_legacy_cache(past)
                else:
                    model_inputs["past_key_values"] = NeuralOpsHybridHFCache()
            return model_inputs

        model.prepare_inputs_for_generation = _patched_prep
        model._neuralops_hf_cache_bridge = True

    return model


HFCacheBridge = NeuralOpsHybridHFCache
PolymorphicGenerationCacheBridge = NeuralOpsHybridHFCache

__all__ = [
    "HFCacheBridge",
    "PolymorphicGenerationCacheBridge",
    "NeuralOpsHybridHFCache",
    "RadixTreeNode",
    "SGLangRadixStateAdapter",
    "_compute_prefix_hash",
    "validate_cache_format",
    "wrap_model_for_generation",
]
