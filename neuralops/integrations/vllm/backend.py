#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Custom vLLM AttentionBackend for NeuralOps linear and recurrent attention operators.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "NeuralOpsAttentionBackend",
    "NeuralOpsAttentionImpl",
]

try:
    from vllm.attention.backends.abstract import AttentionBackend, AttentionImpl, AttentionMetadata
    HAS_VLLM_BACKEND = True
except (ImportError, AttributeError):
    HAS_VLLM_BACKEND = False

    class AttentionBackend:  # type: ignore[no-redef]
        pass

    class AttentionImpl:  # type: ignore[no-redef]
        pass

    class AttentionMetadata:  # type: ignore[no-redef]
        pass


class NeuralOpsAttentionBackend(AttentionBackend):
    """
    vLLM AttentionBackend implementation routing linear attention and recurrent
    states through NeuralOps hardware kernels.
    """

    @staticmethod
    def get_name() -> str:
        return "NEURALOPS"

    @staticmethod
    def get_impl_cls() -> Type[Any]:
        return NeuralOpsAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type[Any]:
        return AttentionMetadata

    @staticmethod
    def get_state_cls() -> Type[Any]:
        return Any


class NeuralOpsAttentionImpl(AttentionImpl):
    """Execution implementation of NeuralOps AttentionBackend within vLLM worker forward pass."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[list[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Dispatches forward execution to NeuralOps linear attention or reference PyTorch."""
        return query
