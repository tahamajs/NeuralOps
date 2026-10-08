"""vLLM and TGI serving integration adapters."""
from __future__ import annotations
from typing import Any
import torch.nn as nn

__all__ = ["VLLMAttentionAdapter", "TGIAttentionAdapter"]


class VLLMAttentionAdapter:
    """Adapts NeuralOps attention modules for vLLM block-manager and forward runners."""

    def __init__(self, attention_type: str, head_dim: int, num_heads: int) -> None:
        self.attention_type = attention_type
        self.head_dim = head_dim
        self.num_heads = num_heads

    def format_metadata(self, block_tables: list[list[int]], context_lens: list[int]) -> dict[str, Any]:
        """Formats sequence metadata into vLLM engine-compatible structures."""
        return {
            "block_tables": block_tables,
            "context_lens": context_lens,
            "max_context_len": max(context_lens) if context_lens else 0,
        }


class TGIAttentionAdapter:
    """Adapts attention modules for Text Generation Inference (TGI) batch execution."""

    def __init__(self, max_batch_size: int = 32) -> None:
        self.max_batch_size = max_batch_size

    def prepare_batch(self, batch_inputs: list[Any]) -> list[Any]:
        return batch_inputs[: self.max_batch_size]
