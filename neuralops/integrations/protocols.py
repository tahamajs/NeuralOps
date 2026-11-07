#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Standardized Structural Subtyping Protocols for Inference & Serving Frameworks.

Defines runtime-checkable protocol contracts across high-throughput serving engines,
paged virtual memory managers, continuous iteration batchers, and prefix caches.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable
import torch


@runtime_checkable
class ServingEngineProtocol(Protocol):
    """Abstract interface defining the execution lifecycle of a serving engine."""

    def initialize(self, *args: Any, **kwargs: Any) -> Any:
        """Initialize engine resources, model weights, and execution context."""
        ...

    def prefill(
        self,
        input_ids: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute prefill phase for prompt sequences and populate KV/recurrent state."""
        ...

    def decode_step(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a single autoregressive decoding step across active sequence slots."""
        ...

    def shutdown(self) -> None:
        """Release allocated execution resources, GPU buffers, and worker contexts."""
        ...


@runtime_checkable
class PagedKVCacheProtocol(Protocol):
    """Abstract interface for non-contiguous paged physical memory management."""

    def allocate_sequence(self, seq_id: int) -> None:
        """Register a new sequence with an empty physical page allocation."""
        ...

    def fork_sequence(self, parent_seq_id: int, child_seq_id: int) -> List[int]:
        """Perform zero-copy sequence forking via virtual block table reference tracking."""
        ...

    def copy_on_write(self, seq_id: int, block_idx: int) -> int:
        """Allocate private physical block when mutating a shared block frame."""
        ...

    def free_sequence(self, seq_id: int) -> None:
        """Reclaim physical page blocks and update allocation reference tables."""
        ...

    def append_token(
        self,
        seq_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Append token key/value states into sequence page tables."""
        ...

    def get_kv(self, seq_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve full sequence key and value representations."""
        ...


@runtime_checkable
class ContinuousBatcherProtocol(Protocol):
    """Abstract interface for request slot management and iteration-level scheduling."""

    def add_request(self, *args: Any, **kwargs: Any) -> None:
        """Enqueue an incoming generation request into the pending scheduling queue."""
        ...

    def step(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        """Advance the continuous batch by executing one prefill or decode iteration step."""
        ...


@runtime_checkable
class RadixTrieProtocol(Protocol):
    """Abstract interface for prefix tree search, caching, and state retrieval."""

    def insert_prefix(
        self,
        token_ids: Union[List[int], torch.Tensor],
        kv_cache: Any,
    ) -> Any:
        """Insert a token sequence and its cached states into the prefix radix trie."""
        ...

    def match_prefix(
        self,
        token_ids: Union[List[int], torch.Tensor],
    ) -> Tuple[int, Optional[Any]]:
        """Find the longest matching prefix tokens and retrieve associated state."""
        ...

    def evict_lru(self, *args: Any, **kwargs: Any) -> int:
        """Evict unreferenced leaf nodes under memory pressure according to LRU policy."""
        ...


__all__ = [
    "ServingEngineProtocol",
    "PagedKVCacheProtocol",
    "ContinuousBatcherProtocol",
    "RadixTrieProtocol",
]
