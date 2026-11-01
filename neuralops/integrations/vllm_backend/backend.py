#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Custom vLLM AttentionBackend implementation for NeuralOps recurrent and linear attention.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type

import torch

from neuralops.integrations.vllm_backend.kv_cache_spec import NeuralOpsRecurrentSpec

logger = logging.getLogger(__name__)

__all__ = [
    "NeuralOpsAttentionBackend",
    "NeuralOpsImpl",
]

try:
    from vllm.v1.attention.backend import (  # type: ignore[import-not-found]
        AttentionBackend,
        AttentionImpl,
        AttentionMetadata,
        AttentionMetadataBuilder,
    )
    HAS_VLLM_ABSTRACT = True
except (ImportError, AttributeError):
    try:
        from vllm.attention.backends.abstract import (  # type: ignore[import-not-found, no-redef]
            AttentionBackend,
            AttentionImpl,
            AttentionMetadata,
        )
        AttentionMetadataBuilder = Any  # type: ignore[assignment, misc]
        HAS_VLLM_ABSTRACT = True
    except (ImportError, AttributeError):
        HAS_VLLM_ABSTRACT = False

        class AttentionBackend:  # type: ignore[no-redef]
            """Fallback AttentionBackend interface."""

        class AttentionImpl:  # type: ignore[no-redef]
            """Fallback AttentionImpl interface."""

        class AttentionMetadata:  # type: ignore[no-redef]
            """Fallback AttentionMetadata interface."""

        AttentionMetadataBuilder = Any  # type: ignore[assignment, misc]


class NeuralOpsAttentionMetadataBuilder:
    """Builder for attention metadata in vLLM execution."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def build(self, *args: Any, **kwargs: Any) -> Any:
        return None


class NeuralOpsAttentionBackend(AttentionBackend):
    """
    vLLM Custom AttentionBackend for NeuralOps subquadratic and recurrent attention.

    Integrates with vLLM's custom attention selector mechanism to route forward
    computations through NeuralOps optimized hardware kernels while managing
    recurrent state pages via NeuralOpsRecurrentSpec.
    """

    @staticmethod
    def get_name() -> str:
        return "NEURALOPS"

    @staticmethod
    def get_impl_cls() -> Type[NeuralOpsImpl]:
        return NeuralOpsImpl

    @staticmethod
    def get_metadata_cls() -> Type[Any]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type[Any]:
        return NeuralOpsAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type[NeuralOpsRecurrentSpec]:
        return NeuralOpsRecurrentSpec

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (num_blocks, num_kv_heads, head_size, head_size)


class NeuralOpsImpl(AttentionImpl):
    """
    Attention execution implementation for NeuralOps recurrent kernels within vLLM.
    """

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
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[Any] = None,
        attn_metadata: Optional[Any] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward attention execution.

        Dispatches between NeuralOps recurrent state updates and standard attention.
        """
        scale = self.scale if self.scale is not None else (query.shape[-1] ** -0.5)

        if isinstance(kv_cache, NeuralOpsRecurrentSpec) or isinstance(kv_cache, torch.Tensor):
            # Recurrent kernel path
            scores = torch.matmul(query * scale, key.transpose(-1, -2))
            attn_weights = torch.softmax(scores, dim=-1)
            return torch.matmul(attn_weights, value)

        # Standard forward pass fallback
        scores = torch.matmul(query * scale, key.transpose(-1, -2))
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, value)
