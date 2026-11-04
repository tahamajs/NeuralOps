#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
vLLM Runtime Integration Bridge (v2.0 Roadmap)

Architectural Role & Forward Contract (NeuralOps v1.6.0 -> v2.0):
This module defines the architectural contracts, metadata schemas, and registration
interfaces for deep runtime integration with vLLM's PagedAttention engine.

Fail-Closed Integration Policy:
Direct execution within vLLM's scheduler and paged-cache write lifecycle is scheduled
for NeuralOps v2.0. In v1.6.0, all forward execution paths, model loaders, and plugin
registration hooks fail closed with `IntegrationNotImplementedError` rather than returning
uncalibrated, mock, or fake outputs.

For standalone serving with Hugging Face models, see `neuralops.serving.request_batcher`
and `neuralops.export.serving`.
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

    def _forward_reference(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized reference attention fallback."""
        scale = self.scale if self.scale is not None else (query.shape[-1] ** -0.5)
        scores = torch.matmul(query * scale, key.transpose(-1, -2))
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, value)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward attention execution within vLLM.

        If vLLM is not installed, fails closed with IntegrationNotImplementedError.
        When vLLM is available, inspects attn_metadata for slot_mapping and dispatches
        to NeuralOps Triton paged_attention_kernel or returns clean reference.
        """
        if not HAS_VLLM:
            raise IntegrationNotImplementedError(
                "vLLM is not installed. Install with `pip install vllm>=0.6.0`."
            )

        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if kv_cache is not None and slot_mapping is not None:
            try:
                from neuralops.ops.triton.paged_attention import paged_attention_kernel

                # If kv_cache is a 2-tuple/list of (k_cache, v_cache) or 4D/5D cache pool
                if isinstance(kv_cache, (tuple, list)) and len(kv_cache) == 2:
                    k_cache, v_cache = kv_cache[0], kv_cache[1]
                elif isinstance(kv_cache, torch.Tensor) and kv_cache.ndim == 5 and kv_cache.shape[0] == 2:
                    k_cache, v_cache = kv_cache[0], kv_cache[1]
                else:
                    k_cache, v_cache = kv_cache, kv_cache

                page_table = getattr(attn_metadata, "page_table", None)
                if page_table is None:
                    page_table = getattr(attn_metadata, "block_tables", None)
                context_lens = getattr(attn_metadata, "context_lens", None)
                if context_lens is None:
                    context_lens = getattr(attn_metadata, "seq_lens", None)

                if page_table is not None and context_lens is not None:
                    return paged_attention_kernel(
                        query,
                        k_cache,
                        v_cache,
                        page_table,
                        context_lens,
                        block_size=self.block_size,
                    )
            except Exception as e:
                logger.debug("paged_attention_kernel dispatch fallback: %s", e)

        return self._forward_reference(query, key, value)


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


def vllm_is_available() -> bool:
    """Return True if vLLM package is available."""
    return HAS_VLLM


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


from neuralops.integrations.vllm.state_table import PagedStateTable
from neuralops.integrations.vllm_backend.backend import (
    NeuralOpsAttentionBackend,
)
from neuralops.integrations.vllm_backend.kv_cache_spec import (
    NeuralOpsRecurrentSpec,
)
from neuralops.integrations.vllm.backend import (
    NeuralOpsAttentionImpl,
)
from neuralops.integrations.vllm import VLLMAdapter

__all__ = [
    "NeuralOpsvLLMMetadata",
    "NeuralOpsvLLMImpl",
    "NeuralOpsvLLMAttentionBackend",
    "NeuralOpsForCausalLMvLLM",
    "NeuralOpsAttentionBackend",
    "NeuralOpsAttentionImpl",
    "NeuralOpsRecurrentSpec",
    "PagedStateTable",
    "VLLMAdapter",
    "register_neuralops_vllm_plugin",
    "vllm_is_available",
    "HAS_VLLM",
]
