#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
RadixRecurrentTree for SGLang Serving Runtime

Combines RadixAttention LRU tree matching with constant O(1) recurrent state reuse.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "RadixRecurrentNode",
    "RadixRecurrentTree",
]


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
        self.access_time: float = time.time()

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class RadixRecurrentTree:
    """
    Radix Tree with LRU eviction tracking token prefixes and their corresponding
    recurrent attention states for SGLang / high-throughput serving.
    """

    def __init__(self) -> None:
        self.root = RadixRecurrentNode(token_chunk=[])
        self.node_count: int = 1

    def match_prefix(self, tokens: Sequence[int]) -> Tuple[List[int], Optional[Any]]:
        """Find the longest matching prefix for tokens in the radix tree.

        Returns:
            Tuple of (matched_token_list, recurrent_state_at_end_of_prefix).
        """
        curr = self.root
        matched_tokens: list[int] = []
        token_idx = 0
        n_tokens = len(tokens)
        last_valid_state: Optional[Any] = None

        while token_idx < n_tokens:
            first_tok = tokens[token_idx]
            if first_tok not in curr.children:
                break

            child = curr.children[first_tok]
            child.access_time = time.time()

            # Compare chunk
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
                if child.recurrent_state is not None:
                    last_valid_state = child.recurrent_state
            else:
                # Partial match inside chunk
                break

        return matched_tokens, last_valid_state

    def insert(
        self,
        tokens: Sequence[int],
        recurrent_state: Optional[Any] = None,
    ) -> RadixRecurrentNode:
        """Insert a token sequence with its associated recurrent state."""
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
                    return leaf
                else:
                    split_node.recurrent_state = recurrent_state
                    return split_node

        curr.recurrent_state = recurrent_state
        return curr

    def _collect_leaves(self, node: RadixRecurrentNode, leaves: list[RadixRecurrentNode]) -> None:
        if node.is_leaf and node is not self.root:
            leaves.append(node)
        for child in node.children.values():
            self._collect_leaves(child, leaves)

    def evict(self, num_nodes: int = 1) -> int:
        """Evict oldest LRU leaf nodes from the tree."""
        evicted = 0
        for _ in range(num_nodes):
            leaves: list[RadixRecurrentNode] = []
            self._collect_leaves(self.root, leaves)
            if not leaves:
                break
            oldest = min(leaves, key=lambda n: n.access_time)
            parent = oldest.parent
            if parent and oldest.token_chunk:
                first_tok = oldest.token_chunk[0]
                if first_tok in parent.children:
                    del parent.children[first_tok]
                    self.node_count -= 1
                    evicted += 1
        return evicted

    def __len__(self) -> int:
        return self.node_count
