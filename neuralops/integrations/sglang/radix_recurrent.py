#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
RadixRecurrentTree for SGLang Serving Runtime

Combines RadixAttention LRU tree matching with constant O(1) recurrent state reuse,
multi-tenant reference counting, and memory bounds management.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "RadixRecurrentNode",
    "RadixRecurrentTree",
    "linear_recurrent_step",
]


def _compute_state_bytes(state: Any) -> int:
    """Compute physical memory footprint in bytes for a cached recurrent state."""
    if state is None:
        return 0
    if isinstance(state, torch.Tensor):
        return state.element_size() * state.nelement()
    if isinstance(state, (list, tuple)):
        return sum(_compute_state_bytes(s) for s in state)
    if isinstance(state, dict):
        return sum(_compute_state_bytes(v) for v in state.values())
    return 0


def linear_recurrent_step(
    state_prev: torch.Tensor,
    k_t: torch.Tensor,
    v_t: torch.Tensor,
    gamma: Union[float, torch.Tensor] = 1.0,
) -> torch.Tensor:
    r"""
    Execute exact linear recurrent state progression step:
    $$S_t = \gamma S_{t-1} + K_t^T V_t$$

    Args:
        state_prev: Previous recurrent state of shape [..., H, D_k, D_v] or [H, D_k, D_v].
        k_t: Key tensor at timestep t of shape [..., H, D_k] or [..., H, 1, D_k].
        v_t: Value tensor at timestep t of shape [..., H, D_v] or [..., H, 1, D_v].
        gamma: Decay factor ($\gamma \in (0, 1]$ or tensor).

    Returns:
        Updated state tensor $S_t$.
    """
    if k_t.ndim == state_prev.ndim - 1:
        # e.g., k_t is [H, D_k] and state_prev is [H, D_k, D_v]
        k_vec = k_t.unsqueeze(-1)  # [H, D_k, 1]
        v_vec = v_t.unsqueeze(-2)  # [H, 1, D_v]
    elif k_t.ndim == state_prev.ndim:
        # e.g. [H, 1, D_k] or [B, H, 1, D_k]
        if k_t.shape[-2] == 1:
            k_vec = k_t.transpose(-1, -2)  # [..., D_k, 1]
            v_vec = v_t  # [..., 1, D_v]
        else:
            k_vec = k_t.unsqueeze(-1)
            v_vec = v_t.unsqueeze(-2)
    else:
        k_vec = k_t.unsqueeze(-1)
        v_vec = v_t.unsqueeze(-2)

    delta = torch.matmul(k_vec, v_vec)
    if isinstance(gamma, torch.Tensor):
        return gamma * state_prev + delta
    if gamma == 1.0:
        return state_prev + delta
    return gamma * state_prev + delta


class RadixRecurrentNode:
    """A node in the RadixRecurrentTree holding a token chunk and its accumulated recurrent state."""

    def __init__(
        self,
        token_chunk: Sequence[int],
        parent: Optional[RadixRecurrentNode] = None,
        recurrent_state: Optional[Any] = None,
    ) -> None:
        self.token_chunk: list[int] = list(token_chunk)
        self.parent = parent
        self.children: Dict[int, RadixRecurrentNode] = {}
        self.recurrent_state = recurrent_state
        self.ref_count: int = 0
        self.access_time: float = time.time()

    @property
    def token_id(self) -> Optional[int]:
        """Convenience accessor for single-token nodes or first token in chunk."""
        return self.token_chunk[0] if self.token_chunk else None

    @property
    def state_tensor(self) -> Optional[torch.Tensor]:
        """Convenience accessor for recurrent state tensor [H, D_k, D_v]."""
        if isinstance(self.recurrent_state, torch.Tensor):
            return self.recurrent_state
        return None

    @property
    def last_accessed_timestamp(self) -> float:
        """Alias for access_time."""
        return self.access_time

    @last_accessed_timestamp.setter
    def last_accessed_timestamp(self, value: float) -> None:
        self.access_time = value

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def state_size_bytes(self) -> int:
        return _compute_state_bytes(self.recurrent_state)


class RadixRecurrentTree:
    """
    Radix Tree with LRU eviction tracking token prefixes and their corresponding
    recurrent attention states for SGLang / high-throughput serving.

    Supports thread-safe multi-tenant access, reference-count pinning,
    LRU eviction, and memory-bounded state management.
    """

    def __init__(
        self,
        max_cache_bytes: Optional[int] = None,
        max_cached_nodes: Optional[int] = None,
    ) -> None:
        self.root = RadixRecurrentNode(token_chunk=[])
        self.node_count: int = 1
        self.max_cache_bytes = max_cache_bytes
        self.max_cached_nodes = max_cached_nodes
        self._lock = threading.RLock()

    @property
    def total_cache_bytes(self) -> int:
        """Calculate total memory consumed by cached recurrent states."""
        total = 0
        with self._lock:
            stack = [self.root]
            while stack:
                node = stack.pop()
                total += node.state_size_bytes
                stack.extend(node.children.values())
        return total

    def match_prefix(
        self,
        tokens: Sequence[int],
    ) -> Tuple[List[int], Optional[Any]]:
        """Find the longest matching prefix for tokens in the radix tree.

        Returns:
            Tuple of (matched_token_list, recurrent_state_at_end_of_prefix).
        """
        matched_tokens, state, _ = self._match_prefix_internal(tokens)
        return matched_tokens, state

    def match_prefix_node(
        self,
        token_ids: Sequence[int],
    ) -> Tuple[RadixRecurrentNode, int]:
        """Find the longest matching node and matched length in tokens.

        Returns:
            Tuple of (matched_node, matched_length).
        """
        with self._lock:
            _, _, node, matched_len = self._match_prefix_internal_full(token_ids)
            return node, matched_len

    def _match_prefix_internal_full(
        self,
        tokens: Sequence[int],
    ) -> Tuple[List[int], Optional[Any], RadixRecurrentNode, int]:
        with self._lock:
            curr = self.root
            matched_tokens: list[int] = []
            token_idx = 0
            n_tokens = len(tokens)
            last_valid_state: Optional[Any] = None
            last_matched_node: RadixRecurrentNode = self.root

            while token_idx < n_tokens:
                first_tok = tokens[token_idx]
                if first_tok not in curr.children:
                    break

                child = curr.children[first_tok]
                child.access_time = time.time()
                chunk = child.token_chunk
                match_len = 0
                while (
                    token_idx + match_len < n_tokens
                    and match_len < len(chunk)
                    and tokens[token_idx + match_len] == chunk[match_len]
                ):
                    match_len += 1

                matched_tokens.extend(chunk[:match_len])
                token_idx += match_len

                if match_len == len(chunk):
                    curr = child
                    last_matched_node = child
                    if child.recurrent_state is not None:
                        last_valid_state = child.recurrent_state
                else:
                    break

            return matched_tokens, last_valid_state, last_matched_node, token_idx

    def _match_prefix_internal(
        self,
        tokens: Sequence[int],
    ) -> Tuple[List[int], Optional[Any], int]:
        matched_tokens, state, _, matched_len = self._match_prefix_internal_full(tokens)
        return matched_tokens, state, matched_len

    def match_and_step(
        self,
        token_ids: Sequence[int],
        layer_idx: Optional[int] = None,
    ) -> Tuple[List[int], Optional[Any], List[int]]:
        """
        Matches longest cached token prefix and returns:
        (matched_tokens, cached_recurrent_state, remaining_tokens)

        Args:
            token_ids: Full sequence of token IDs to match against cached prefixes.
            layer_idx: Optional layer index if recurrent states are partitioned per-layer.

        Returns:
            Tuple of:
                matched_tokens: Prefix token IDs found in cache.
                cached_recurrent_state: Recurrent state tensor (or state dict/tuple)
                                        corresponding to the end of matched_tokens.
                remaining_tokens: Suffix tokens requiring active prefill.
        """
        tokens_list = list(token_ids)
        matched_tokens, state = self.match_prefix(tokens_list)
        remaining_tokens = tokens_list[len(matched_tokens):]

        if layer_idx is not None and state is not None:
            if isinstance(state, (list, tuple)) and layer_idx < len(state):
                state = state[layer_idx]
            elif isinstance(state, dict) and layer_idx in state:
                state = state[layer_idx]

        return matched_tokens, state, remaining_tokens

    def insert(
        self,
        tokens: Sequence[int],
        recurrent_state: Optional[Any] = None,
    ) -> RadixRecurrentNode:
        """Insert a token sequence with its associated recurrent state."""
        with self._lock:
            # Check capacity limits before insertion
            self._ensure_memory_bounds()

            curr = self.root
            tokens_list = list(tokens)
            token_idx = 0
            n_tokens = len(tokens_list)

            while token_idx < n_tokens:
                first_tok = tokens_list[token_idx]
                if first_tok not in curr.children:
                    # Create a new leaf child with the remaining tokens
                    new_node = RadixRecurrentNode(
                        token_chunk=tokens_list[token_idx:],
                        parent=curr,
                        recurrent_state=recurrent_state,
                    )
                    curr.children[first_tok] = new_node
                    self.node_count += 1
                    self._ensure_memory_bounds()
                    return new_node

                child = curr.children[first_tok]
                child.access_time = time.time()
                chunk = child.token_chunk

                # Find common prefix length between chunk and remaining tokens
                match_len = 0
                while (
                    token_idx + match_len < n_tokens
                    and match_len < len(chunk)
                    and tokens_list[token_idx + match_len] == chunk[match_len]
                ):
                    match_len += 1

                if match_len == len(chunk):
                    # Full chunk matched, continue traversal
                    token_idx += match_len
                    curr = child
                else:
                    # Split existing child node
                    split_node = RadixRecurrentNode(
                        token_chunk=chunk[:match_len],
                        parent=curr,
                        recurrent_state=None,
                    )
                    split_node.ref_count = child.ref_count
                    # Re-parent old child
                    child.token_chunk = chunk[match_len:]
                    child.parent = split_node
                    split_node.children[child.token_chunk[0]] = child

                    curr.children[first_tok] = split_node
                    self.node_count += 1

                    token_idx += match_len
                    if token_idx < n_tokens:
                        # Add new leaf for remainder
                        leaf = RadixRecurrentNode(
                            token_chunk=tokens_list[token_idx:],
                            parent=split_node,
                            recurrent_state=recurrent_state,
                        )
                        split_node.children[tokens_list[token_idx]] = leaf
                        self.node_count += 1
                        self._ensure_memory_bounds()
                        return leaf
                    else:
                        split_node.recurrent_state = recurrent_state
                        self._ensure_memory_bounds()
                        return split_node

            curr.recurrent_state = recurrent_state
            self._ensure_memory_bounds()
            return curr

    def inc_ref(self, tokens: Sequence[int]) -> int:
        """Increment pin reference count along the prefix path for active serving tenants."""
        with self._lock:
            curr = self.root
            tokens_list = list(tokens)
            token_idx = 0
            n_tokens = len(tokens_list)
            pinned_nodes = 0

            while token_idx < n_tokens:
                first_tok = tokens_list[token_idx]
                if first_tok not in curr.children:
                    break
                child = curr.children[first_tok]
                chunk_len = len(child.token_chunk)
                if tokens_list[token_idx : token_idx + chunk_len] == child.token_chunk:
                    child.ref_count += 1
                    pinned_nodes += 1
                    token_idx += chunk_len
                    curr = child
                else:
                    break
            return pinned_nodes

    def dec_ref(self, tokens: Sequence[int]) -> int:
        """Decrement pin reference count along the prefix path when a request completes."""
        with self._lock:
            curr = self.root
            tokens_list = list(tokens)
            token_idx = 0
            n_tokens = len(tokens_list)
            unpinned_nodes = 0

            while token_idx < n_tokens:
                first_tok = tokens_list[token_idx]
                if first_tok not in curr.children:
                    break
                child = curr.children[first_tok]
                chunk_len = len(child.token_chunk)
                if tokens_list[token_idx : token_idx + chunk_len] == child.token_chunk:
                    child.ref_count = max(0, child.ref_count - 1)
                    unpinned_nodes += 1
                    token_idx += chunk_len
                    curr = child
                else:
                    break
            return unpinned_nodes

    def _collect_leaves(self, node: RadixRecurrentNode, leaves: list[RadixRecurrentNode]) -> None:
        if node.is_leaf and node is not self.root:
            leaves.append(node)
        for child in node.children.values():
            self._collect_leaves(child, leaves)

    def evict(self, num_nodes: int = 1) -> int:
        """
        Evict oldest LRU unpinned leaf nodes from the tree.
        Pinned nodes (ref_count > 0) are protected against eviction.
        """
        with self._lock:
            evicted = 0
            for _ in range(num_nodes):
                leaves: list[RadixRecurrentNode] = []
                self._collect_leaves(self.root, leaves)
                unpinned_leaves = [n for n in leaves if n.ref_count == 0]
                if not unpinned_leaves:
                    break
                oldest = min(unpinned_leaves, key=lambda n: n.access_time)
                parent = oldest.parent
                if parent and oldest.token_chunk:
                    first_tok = oldest.token_chunk[0]
                    if first_tok in parent.children:
                        del parent.children[first_tok]
                        oldest.recurrent_state = None
                        self.node_count -= 1
                        evicted += 1
            return evicted

    def evict_lru(self, num_bytes_to_free: int = 0) -> int:
        """
        Traverses leaf nodes and purges least recently used states until
        at least num_bytes_to_free bytes have been freed (or all unpinned leaves evicted).
        If num_bytes_to_free <= 0, evicts at least 1 LRU node if available.

        Returns:
            Total bytes freed by eviction.
        """
        with self._lock:
            bytes_freed = 0
            while True:
                leaves: list[RadixRecurrentNode] = []
                self._collect_leaves(self.root, leaves)
                unpinned_leaves = [n for n in leaves if n.ref_count == 0]
                if not unpinned_leaves:
                    break
                oldest = min(unpinned_leaves, key=lambda n: n.access_time)
                freed_here = oldest.state_size_bytes
                parent = oldest.parent
                if parent and oldest.token_chunk:
                    first_tok = oldest.token_chunk[0]
                    if first_tok in parent.children:
                        del parent.children[first_tok]
                        oldest.recurrent_state = None
                        self.node_count -= 1
                        bytes_freed += freed_here

                if num_bytes_to_free <= 0:
                    break
                if bytes_freed >= num_bytes_to_free:
                    break
            return bytes_freed

    def _ensure_memory_bounds(self) -> None:
        """Evict unpinned leaf nodes until memory bytes and node limits are satisfied."""
        if self.max_cached_nodes is not None:
            while self.node_count > self.max_cached_nodes:
                freed = self.evict(1)
                if freed == 0:
                    break

        if self.max_cache_bytes is not None:
            while self.total_cache_bytes > self.max_cache_bytes:
                freed = self.evict(1)
                if freed == 0:
                    break

    def __len__(self) -> int:
        with self._lock:
            return self.node_count
