#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Native vLLM Integration Plugin

Registers custom attention backends, kernels, and model architectures within vLLM.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Optional
import torch
import torch.nn as nn

from neuralops.core.errors import IntegrationNotImplementedError

logger = logging.getLogger("NeuralOps.vLLM")

try:
    import vllm
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False


@dataclass
class NeuralOpsvLLMMetadata:
    """Metadata for NeuralOps linear and hybrid paged attention execution in vLLM."""
    num_prefill_tokens: int
    num_decode_tokens: int
    seq_ids: list[int]
    page_table: torch.Tensor
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    is_prompt: bool
    max_context_len: int = 0


class NeuralOpsvLLMImpl:
    """Attention implementation executed during vLLM forward passes."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[list[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        block_size: int = 16,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.block_size = block_size
        self.sliding_window = sliding_window

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Fail closed until vLLM cache writes and metadata are implemented."""
        raise IntegrationNotImplementedError(
            "NeuralOps vLLM attention execution is not yet compatible with vLLM's "
            "paged-cache write and scheduler metadata contract; no approximate output is returned. "
            "See docs/roadmap/VLLM_PLAN.md for details."
        )


class NeuralOpsvLLMAttentionBackend:
    """Registered vLLM AttentionBackend for NeuralOps."""

    @staticmethod
    def get_name() -> str:
        return "NEURALOPS_HYBRID"

    @staticmethod
    def get_impl_cls() -> type[NeuralOpsvLLMImpl]:
        return NeuralOpsvLLMImpl

    @staticmethod
    def get_metadata_cls() -> type[NeuralOpsvLLMMetadata]:
        return NeuralOpsvLLMMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        """Returns physical state pool shape: [num_blocks, num_kv_heads, head_size, head_size]."""
        return (num_blocks, num_kv_heads, head_size, head_size)


class NeuralOpsForCausalLMvLLM(nn.Module):
    """Reserved model-loader symbol; checkpoint-backed vLLM loading is not implemented."""

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__()
        raise IntegrationNotImplementedError(
            "This placeholder cannot reconstruct a checkpoint and must not initialize random weights. "
            "Load the base model with Transformers and export it using neuralops.export instead. "
            "See docs/roadmap/VLLM_PLAN.md for details."
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        kv_caches: Optional[list[torch.Tensor]] = None,
        attn_metadata: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise IntegrationNotImplementedError(
            "Checkpoint-backed vLLM model execution is not implemented. "
            "See docs/roadmap/VLLM_PLAN.md for details."
        )

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: Any = None) -> torch.Tensor:
        raise IntegrationNotImplementedError(
            "Checkpoint-backed vLLM model execution is not implemented. "
            "See docs/roadmap/VLLM_PLAN.md for details."
        )

    def load_weights(self, weights: Any) -> None:
        """Reject vLLM weight loading until checkpoint key mapping is implemented."""
        raise IntegrationNotImplementedError(
            "vLLM checkpoint weight mapping is not implemented. "
            "See docs/roadmap/VLLM_PLAN.md for details."
        )


def register_neuralops_vllm_plugin() -> bool:
    """Report whether the optional vLLM integration is safe to register."""
    warnings.warn(
        "vLLM integration is experimental/stub in v1.6.0 and deferred to v2.0. "
        "See docs/roadmap/VLLM_PLAN.md for details.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not HAS_VLLM:
        logger.debug("vLLM is not installed; plugin registration skipped.")
        return False
    logger.warning(
        "NeuralOps vLLM plugin was not registered: backend lifecycle, cache writes, "
        "and checkpoint mapping have not passed compatibility verification."
    )
    return False


__all__ = [
    "NeuralOpsvLLMMetadata",
    "NeuralOpsvLLMImpl",
    "NeuralOpsvLLMAttentionBackend",
    "NeuralOpsForCausalLMvLLM",
    "register_neuralops_vllm_plugin",
    "HAS_VLLM",
]
