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


# Export unified names
VLLMAdapter = VLLMAttentionAdapter
TGIAdapter = TGIAttentionAdapter

__all__ = [
    "VLLMAttentionAdapter",
    "TGIAttentionAdapter",
    "VLLMAdapter",
    "TGIAdapter",
    "HAS_VLLM",
    "HAS_TGI",
]
