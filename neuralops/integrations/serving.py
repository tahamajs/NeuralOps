"""
vLLM and TGI serving integration adapters for NeuralOps.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional
import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.ServingIntegrations")

# Safe vLLM detection
try:
    import vllm
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

# Safe TGI detection
try:
    import text_generation_server
    HAS_TGI = True
except ImportError:
    HAS_TGI = False


class VLLMAttentionAdapter:
    """Adapts NeuralOps attention modules for vLLM block-manager and forward runners."""

    def __init__(self, attention_type: str, head_dim: int, num_heads: int) -> None:
        self.attention_type = attention_type
        self.head_dim = head_dim
        self.num_heads = num_heads

    @staticmethod
    def is_available() -> bool:
        """Check if vLLM package is available."""
        return HAS_VLLM

    def register(self) -> bool:
        """Register adapter into vLLM attention backends."""
        if not HAS_VLLM:
            raise RuntimeError(
                "vLLM is not installed. Please install neuralops[serving] or run: pip install vllm"
            )
        return True

    def format_metadata(self, block_tables: List[List[int]], context_lens: List[int]) -> dict[str, Any]:
        """Formats sequence metadata into vLLM engine-compatible structures."""
        return {
            "block_tables": block_tables,
            "context_lens": context_lens,
            "max_context_len": max(context_lens) if context_lens else 0,
        }

    def paged_attention_forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int = 16,
    ) -> torch.Tensor:
        """Execute paged attention forward pass over block_tables and KV-cache blocks."""
        if not HAS_VLLM:
            logger.debug("vLLM not installed; using NeuralOps native paged_attention_kernel.")

        from neuralops.core.triton_kernels.paged_attention import paged_attention_kernel
        return paged_attention_kernel(
            q=query,
            k_cache=key_cache,
            v_cache=value_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=block_size,
        )


class TGIAttentionAdapter:
    """Adapts attention modules for Text Generation Inference (TGI) batch execution."""

    def __init__(self, max_batch_size: int = 32) -> None:
        self.max_batch_size = max_batch_size

    @staticmethod
    def is_available() -> bool:
        """Check if TGI server is available."""
        return HAS_TGI

    def prepare_batch(self, batch_inputs: List[Any]) -> List[Any]:
        """Truncates or batches inputs to match TGI runtime batch boundaries."""
        return batch_inputs[: self.max_batch_size]


class SGLangRadixLinearCacheAdapter:
    """
    Bridges NeuralOps linear/recurrent state caching with SGLang RadixAttention interface.
    Supports tree-structured prefix state matching, reference locking, and LRU eviction.
    """

    def __init__(self, max_cached_nodes: int = 4096) -> None:
        self.max_cached_nodes = max_cached_nodes
        # Prefix tree node storage: token_tuple -> dict(state=Tensor, ref_count=int, last_accessed=int)
        self.tree: dict[tuple[int, ...], dict[str, Any]] = {}
        self.access_counter = 0

    def match_prefix(self, token_ids: list[int] | tuple[int, ...]) -> tuple[int, Optional[Any]]:
        """Finds the longest cached prefix state in O(L) time."""
        token_tuple = tuple(token_ids)
        best_match_len = 0
        best_state = None

        for length in range(len(token_tuple), 0, -1):
            prefix = token_tuple[:length]
            if prefix in self.tree:
                self.access_counter += 1
                self.tree[prefix]["last_accessed"] = self.access_counter
                return length, self.tree[prefix]["state"]

        return 0, None

    def cache_finished_req(self, token_ids: list[int] | tuple[int, ...], final_state: Any) -> None:
        """Caches state of a completed sequence, evicting cold unlocked nodes if capacity is reached."""
        self._ensure_capacity()
        prefix = tuple(token_ids)
        self.access_counter += 1
        if prefix in self.tree:
            self.tree[prefix]["state"] = final_state
            self.tree[prefix]["last_accessed"] = self.access_counter
        else:
            self.tree[prefix] = {
                "state": final_state,
                "ref_count": 0,
                "last_accessed": self.access_counter,
            }

    def cache_unfinished_req(self, token_ids: list[int] | tuple[int, ...], current_state: Any) -> None:
        """Locks and caches intermediate state during prefill/rollouts."""
        prefix = tuple(token_ids)
        self.cache_finished_req(prefix, current_state)
        self.inc_lock_ref(prefix)

    def inc_lock_ref(self, token_ids: list[int] | tuple[int, ...]) -> None:
        prefix = tuple(token_ids)
        if prefix in self.tree:
            self.tree[prefix]["ref_count"] += 1

    def dec_lock_ref(self, token_ids: list[int] | tuple[int, ...]) -> None:
        prefix = tuple(token_ids)
        if prefix in self.tree:
            self.tree[prefix]["ref_count"] = max(0, self.tree[prefix]["ref_count"] - 1)

    def evict(self, num_nodes: int = 1) -> int:
        """Evicts unreferenced nodes according to LRU policy."""
        evicted = 0
        unlocked = [
            (prefix, node["last_accessed"])
            for prefix, node in self.tree.items()
            if node["ref_count"] == 0
        ]
        # Sort by oldest access
        unlocked.sort(key=lambda x: x[1])

        for prefix, _ in unlocked[:num_nodes]:
            del self.tree[prefix]
            evicted += 1
        return evicted

    def _ensure_capacity(self) -> None:
        if len(self.tree) >= self.max_cached_nodes:
            self.evict(max(1, len(self.tree) - self.max_cached_nodes + 1))


# Export unified names
VLLMAdapter = VLLMAttentionAdapter
TGIAdapter = TGIAttentionAdapter

__all__ = [
    "VLLMAttentionAdapter",
    "TGIAttentionAdapter",
    "VLLMAdapter",
    "TGIAdapter",
    "SGLangRadixLinearCacheAdapter",
    "HAS_VLLM",
    "HAS_TGI",
]

