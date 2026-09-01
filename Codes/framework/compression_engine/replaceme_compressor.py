"""
ReplaceMe: Naive Layer Removal with Linear Replacement
Registered as 'replaceme'
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


@COMPRESSOR_REGISTRY.register("replaceme")
class ReplaceMeCompressor(BaseCompressor):
    """
    Replaces the last N layers of a Transformer encoder with a simple LayerNorm + Linear block.
    """

    def __init__(
        self,
        model: nn.Module,
        remove_layers: int = 2,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, config=config, **kwargs)
        self.remove_layers = remove_layers

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        orig_params = sum(p.numel() for p in self.model.parameters())
        dim = getattr(
            self.model,
            "config",
            None,
        )
        hidden_size = 64
        if dim is not None:
            hidden_size = getattr(dim, "hidden_size", getattr(dim, "dim", 64))

        # Check encoder layers in various architectures (BERT, DistilBERT, etc.)
        layers = None
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            layers = self.model.encoder.layer
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "layer"):
            layers = self.model.transformer.layer
        elif hasattr(self.model, "bert") and hasattr(self.model.bert.encoder, "layer"):
            layers = self.model.bert.encoder.layer

        if layers is not None:
            num_layers = len(layers)
            n_remove = min(self.remove_layers, num_layers)
            for i in range(num_layers - n_remove, num_layers):
                layers[i] = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                )

        compressed_params = sum(p.numel() for p in self.model.parameters())

        stats = {
            "compressor_name": "replaceme",
            "original_parameters": orig_params,
            "compressed_parameters": compressed_params,
            "compression_ratio": orig_params / max(1, compressed_params),
            "layers_removed": self.remove_layers,
        }
        return self.model, stats
