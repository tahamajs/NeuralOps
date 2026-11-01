#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Recurrent State & KV Cache Specification for vLLM Integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

__all__ = ["NeuralOpsRecurrentSpec"]

try:
    from vllm.v1.kv_cache_interface import KVCacheSpec  # type: ignore[import-not-found]
except (ImportError, AttributeError):
    try:
        from vllm.attention.backends.abstract import KVCacheSpec  # type: ignore[import-not-found, no-redef]
    except (ImportError, AttributeError):

        class KVCacheSpec:  # type: ignore[no-redef]
            """Fallback KVCacheSpec abstract interface when vLLM is not present."""

            def __init__(self, block_size: int = 16, **kwargs: Any) -> None:
                self.block_size = block_size


@dataclass(frozen=True)
class NeuralOpsRecurrentSpec(KVCacheSpec):
    """
    Recurrent KV Cache specification for subquadratic and linear attention operators.

    Unlike standard quadratic KV caches which store historical (key, value) token
    sequences of shape (batch, heads, seq_len, head_dim), recurrent attention
    mechanisms maintain fixed-size recurrent state tensors:
        S in R^{num_heads x head_dim x state_dim}
    yielding constant O(1) memory consumption independent of sequence length.
    """

    num_heads_val: int = 32
    head_dim_val: int = 64
    state_dim_val: Optional[int] = None
    dtype: torch.dtype = torch.float16
    block_size: int = 16
    device: str = "cuda"

    @property
    def num_heads(self) -> int:
        return self.num_heads_val

    @property
    def head_dim(self) -> int:
        return self.head_dim_val

    @property
    def state_dim(self) -> int:
        return self.state_dim_val if self.state_dim_val is not None else self.head_dim_val

    @property
    def tokens_per_state(self) -> int:
        return 1

    @property
    def page_size_bytes(self) -> int:
        """Size in bytes of a single state page allocation."""
        dtype_bytes = torch.tensor([], dtype=self.dtype).element_size()
        return self.num_heads * self.head_dim * self.state_dim * dtype_bytes

    @property
    def state_content_size_bytes(self) -> int:
        """Total memory consumed by recurrent state per sequence."""
        return self.page_size_bytes

    def max_memory_usage_bytes(self, vllm_config: Any) -> int:
        """Maximum possible memory usage of recurrent state."""
        return self.page_size_bytes

    def allocate_state(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Allocate zero-initialized recurrent state buffer."""
        dev = device or torch.device(self.device)
        return torch.zeros(
            (batch_size, self.num_heads, self.head_dim, self.state_dim),
            dtype=self.dtype,
            device=dev,
        )
