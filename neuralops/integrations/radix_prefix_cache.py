#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Radix Prefix Cache

Provides a standalone Radix Trie prefix cache manager for subquadratic attention caching.
"""

from __future__ import annotations

import inspect
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.SGLang")


class RadixPrefixNode:
    """
    A node in the Radix Trie representing a shared sequence of token IDs.

    Stores:
    - token_chunk: List of tokens represented contiguously by this node.
    - children: Sub-branches keyed by the first token of the child chunk.
    - kv_cache: The physical KV-cache or recurrent state tensor after processing this chunk.
    - ref_count: Active references (pins) preventing node eviction.
    - last_accessed: Monotonic timestamp for LRU eviction ranking.
    """

    def __init__(self, token_chunk: List[int], kv_cache: Optional[Any] = None):
        self.token_chunk = list(token_chunk)
        self.children: Dict[int, RadixPrefixNode] = {}
        self.kv_cache = kv_cache
        self.ref_count = 0
        self.last_accessed = time.time()
        self.parent: Optional[RadixPrefixNode] = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def total_tokens(self) -> int:
        return len(self.token_chunk)


class SGLangRadixAdapter:
    """
    Radix Tree Prefix Cache Manager for LLM Serving with SGLang-compatible semantics.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Any = None,
        max_cached_tokens: int = 8192,
        max_cached_prefixes: int = 256,
        max_depth: int = 128,
        device: Union[str, torch.device] = "cuda",
    ):
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.model = model.to(self.device).eval() if model is not None else None
        self.tokenizer = tokenizer
        self.max_cached_tokens = max_cached_tokens
        self.max_cached_prefixes = max_cached_prefixes
        self.max_depth = max_depth

        self.root = RadixPrefixNode([])
        self._lock = threading.RLock()

        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        self._total_nodes = 1

    def _ensure_token_list(self, token_ids: Union[List[int], torch.Tensor]) -> List[int]:
        if isinstance(token_ids, torch.Tensor):
            return token_ids.detach().cpu().view(-1).tolist()
        return list(token_ids)

    def insert_prefix(self, token_ids: Union[List[int], torch.Tensor], kv_cache: Any) -> RadixPrefixNode:
        """
        Inserts a token sequence and its corresponding KV cache into the Radix Trie.
        Splits existing nodes if the inserted prefix diverges mid-chunk.
        """
        tokens = self._ensure_token_list(token_ids)
        if not tokens:
            return self.root

        with self._lock:
            while self._total_nodes >= self.max_cached_prefixes:
                freed = self.evict_lru()
                if freed == 0:
                    break

            curr = self.root
            idx = 0

            while idx < len(tokens):
                curr.ref_count += 1
                curr.last_accessed = time.time()
                first_tok = tokens[idx]

                if first_tok not in curr.children:
                    new_node = RadixPrefixNode(tokens[idx:], kv_cache=kv_cache)
                    new_node.parent = curr
                    new_node.ref_count = 1
                    curr.children[first_tok] = new_node
                    self._total_nodes += 1
                    return new_node

                child = curr.children[first_tok]
                chunk = child.token_chunk
                common_len = 0
                while (
                    common_len < len(chunk)
                    and idx + common_len < len(tokens)
                    and chunk[common_len] == tokens[idx + common_len]
                ):
                    common_len += 1

                if common_len == len(chunk):
                    idx += common_len
                    curr = child
                else:
                    split_node = RadixPrefixNode(chunk[:common_len], kv_cache=None)
                    split_node.parent = curr
                    split_node.ref_count = child.ref_count + 1
                    split_node.last_accessed = time.time()
                    curr.children[first_tok] = split_node
                    self._total_nodes += 1

                    child.token_chunk = chunk[common_len:]
                    child.parent = split_node
                    split_node.children[child.token_chunk[0]] = child

                    idx += common_len
                    if idx < len(tokens):
                        new_leaf = RadixPrefixNode(tokens[idx:], kv_cache=kv_cache)
                        new_leaf.parent = split_node
                        new_leaf.ref_count = 1
                        split_node.children[tokens[idx]] = new_leaf
                        self._total_nodes += 1
                        return new_leaf
                    else:
                        split_node.kv_cache = kv_cache
                        return split_node

            curr.ref_count += 1
            curr.kv_cache = kv_cache
            return curr

    def match_prefix(self, token_ids: Union[List[int], torch.Tensor]) -> Tuple[int, Optional[Any]]:
        """
        Finds the longest cached prefix matching the start of token_ids.

        Returns:
            (matched_length, kv_cache) or (0, None) on complete cache miss.
        """
        tokens = self._ensure_token_list(token_ids)
        if not tokens:
            return 0, None

        with self._lock:
            curr = self.root
            idx = 0
            best_matched_len = 0
            best_kv = None

            while idx < len(tokens):
                first_tok = tokens[idx]
                if first_tok not in curr.children:
                    break

                child = curr.children[first_tok]
                chunk = child.token_chunk
                match_count = 0
                while (
                    match_count < len(chunk)
                    and idx + match_count < len(tokens)
                    and chunk[match_count] == tokens[idx + match_count]
                ):
                    match_count += 1

                if match_count == len(chunk):
                    idx += match_count
                    curr = child
                    curr.last_accessed = time.time()
                    if curr.kv_cache is not None:
                        best_matched_len = idx
                        best_kv = curr.kv_cache
                else:
                    break

            if best_matched_len > 0:
                self._hit_count += 1
                return best_matched_len, best_kv
            else:
                self._miss_count += 1
                return 0, None

    def prefill_with_reuse(self, input_ids: torch.Tensor) -> Tuple[Any, torch.Tensor]:
        """
        Runs prefill by retrieving longest cached prefix and executing only the suffix.

        Returns:
            (full_kv_cache, logits_for_next_token)
        """
        if self.model is None:
            raise RuntimeError("Model is required to execute prefill_with_reuse.")

        inp = input_ids.to(self.device)
        tokens = self._ensure_token_list(inp)

        matched_len, cached_kv = self.match_prefix(tokens)

        with torch.no_grad():
            if matched_len > 0 and cached_kv is not None and matched_len < len(tokens):
                suffix = inp[:, matched_len:]
                pos_ids = torch.arange(matched_len, len(tokens), device=inp.device).unsqueeze(0)
                outputs = _call_model(self.model, suffix, past_key_values=cached_kv, position_ids=pos_ids, use_cache=True)
                full_kv = getattr(outputs, "past_key_values", cached_kv)
                logits = outputs.logits[:, -1, :] if hasattr(outputs, "logits") else outputs[:, -1, :]
            elif matched_len == len(tokens) and cached_kv is not None:
                # The cache already includes every prompt token, but does not retain
                # the final-token logits needed for the next decode step.
                outputs = _call_model(self.model, inp, use_cache=True)
                full_kv = getattr(outputs, "past_key_values", None)
                logits = outputs.logits[:, -1, :] if hasattr(outputs, "logits") else outputs[:, -1, :]
            else:
                outputs = _call_model(self.model, inp, use_cache=True)
                full_kv = getattr(outputs, "past_key_values", None)
                logits = outputs.logits[:, -1, :] if hasattr(outputs, "logits") else outputs[:, -1, :]

            self.insert_prefix(tokens, full_kv)

        return full_kv, logits

    def decode_step(self, kv_cache: Any, last_token: int) -> Tuple[Any, int]:
        """Single autoregressive decode step using provided KV state."""
        if self.model is None:
            raise RuntimeError("Model is required to execute decode_step.")

        inp = torch.tensor([[last_token]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = _call_model(self.model, inp, past_key_values=kv_cache, use_cache=True)
            new_kv = getattr(outputs, "past_key_values", kv_cache)
            logits = outputs.logits[:, -1, :] if hasattr(outputs, "logits") else outputs[:, -1, :]
            next_tok = int(torch.argmax(logits, dim=-1).item())

        return new_kv, next_tok

    def free_sequence(self, token_ids: Union[List[int], torch.Tensor]) -> None:
        """Decrements reference counters along a sequence path upon request completion."""
        tokens = self._ensure_token_list(token_ids)
        with self._lock:
            curr = self.root
            idx = 0
            while idx < len(tokens):
                first_tok = tokens[idx]
                if first_tok not in curr.children:
                    break
                child = curr.children[first_tok]
                child.ref_count = max(0, child.ref_count - 1)
                idx += len(child.token_chunk)
                curr = child

    def evict_lru(self) -> int:
        """Evicts the least-recently-accessed unpinned leaf node."""
        with self._lock:
            unpinned_leaves: List[Tuple[float, RadixPrefixNode, int]] = []

            def _find_leaves(node: RadixPrefixNode):
                for first_tok, child in list(node.children.items()):
                    if child.is_leaf and child.ref_count == 0:
                        unpinned_leaves.append((child.last_accessed, child, first_tok))
                    else:
                        _find_leaves(child)

            _find_leaves(self.root)
            if not unpinned_leaves:
                return 0

            unpinned_leaves.sort(key=lambda x: x[0])
            _, target_leaf, first_tok = unpinned_leaves[0]
            parent = target_leaf.parent
            if parent is not None and first_tok in parent.children:
                del parent.children[first_tok]
                target_leaf.kv_cache = None
                self._total_nodes -= 1
                self._eviction_count += 1
                return len(target_leaf.token_chunk)
            return 0

    def stats(self) -> Dict[str, Any]:
        """Returns statistics on prefix trie state, hits, misses, and memory."""
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = (self._hit_count / max(1, total_requests)) * 100.0

            total_tokens = 0
            def _count_tokens(n: RadixPrefixNode):
                nonlocal total_tokens
                total_tokens += len(n.token_chunk)
                for c in n.children.values():
                    _count_tokens(c)

            _count_tokens(self.root)

            return {
                "total_nodes": self._total_nodes,
                "cached_tokens": total_tokens,
                "total_cached_tokens": total_tokens,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "eviction_count": self._eviction_count,
                "hit_rate": round(hit_rate, 2),
                "cache_hit_rate": round(hit_rate, 2),
            }

    def visualize(self) -> str:
        """Renders an ASCII tree representation of cached token prefixes."""
        with self._lock:
            lines = [f"RadixPrefixTree [Nodes: {self._total_nodes}]"]

            def _build(node: RadixPrefixNode, prefix: str = ""):
                for i, (tok, child) in enumerate(node.children.items()):
                    is_last = (i == len(node.children) - 1)
                    branch = "└── " if is_last else "├── "
                    sub_prefix = "    " if is_last else "│   "
                    kv_flag = " [KV]" if child.kv_cache is not None else ""
                    pin_flag = f" (pins: {child.ref_count})"
                    lines.append(f"{prefix}{branch}{child.token_chunk}{kv_flag}{pin_flag}")
                    _build(child, prefix + sub_prefix)

            _build(self.root)
            return "\n".join(lines)


# Canonical alias for standalone Radix tree
RadixTree = SGLangRadixAdapter


def _call_model(model: nn.Module, input_ids: torch.Tensor, **optional_kwargs: Any) -> Any:
    """Pass only supported model kwargs without catching errors raised inside forward."""
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return model(input_ids)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )
    accepted = {
        name: value
        for name, value in optional_kwargs.items()
        if accepts_kwargs or name in signature.parameters
    }
    return model(input_ids, **accepted)


__all__ = [
    "RadixPrefixNode",
    "SGLangRadixAdapter",
    "RadixTree",
]
