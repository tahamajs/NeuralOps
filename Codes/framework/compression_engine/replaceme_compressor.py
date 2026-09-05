"""
ReplaceMe: Layer Removal and Replacement with Linear Residual Block.
Replaces the last N layers of a Transformer encoder or decoder with a compatible
LayerNorm + Linear residual block that conforms to transformer forward calling conventions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


class ReplacementBlock(nn.Module):
    """
    Drop-in replacement for a transformer layer.
    Preserves tensor shapes, handles variable signature inputs (*args, **kwargs),
    and maintains signal propagation via a residual connection.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize to zero so block acts as near-identity at initialization
        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, ...]:
        normed = self.ln(hidden_states)
        out = hidden_states + self.linear(normed)
        return (out,)


@COMPRESSOR_REGISTRY.register("replaceme")
@COMPRESSOR_REGISTRY.register("replaceme_compressor")
class ReplaceMeCompressor(BaseCompressor):
    """
    Replaces the trailing N layers of a Transformer with a lightweight LayerNorm + Linear residual block.
    """

    def __init__(
        self,
        model: nn.Module,
        remove_layers: int = 2,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="replaceme", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", config))
        if cfg is not None:
            remove_layers = getattr(cfg, "remove_layers", remove_layers)
        self.remove_layers = max(1, int(remove_layers))

    def _find_transformer_layers(self) -> Optional[nn.ModuleList]:
        """Finds transformer layers across diverse architectures."""
        candidates = [
            ("encoder", "layer"),
            ("roberta", "encoder", "layer"),
            ("bert", "encoder", "layer"),
            ("distilbert", "transformer", "layer"),
            ("transformer", "layer"),
            ("transformer", "h"),
            ("model", "layers"),
            ("gpt2", "transformer", "h"),
        ]
        for path in candidates:
            curr = self.model
            found = True
            for attr in path:
                if hasattr(curr, attr):
                    curr = getattr(curr, attr)
                else:
                    found = False
                    break
            if found and isinstance(curr, (nn.ModuleList, list)):
                return curr

        # Fallback search for any ModuleList with transformer layer-like children
        for _, module in self.model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 2:
                first = module[0]
                cls_lower = first.__class__.__name__.lower()
                if any(k in cls_lower for k in ["layer", "block", "decoderlayer", "encoderlayer"]):
                    return module
        return None

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        orig_params = self._count_params(self.model)

        # Resolve hidden_size
        hidden_size = 768
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "dim", getattr(cfg, "d_model", 768)))

        layers = self._find_transformer_layers()
        layers_replaced = 0

        if layers is not None:
            num_layers = len(layers)
            n_remove = min(self.remove_layers, num_layers)
            for i in range(num_layers - n_remove, num_layers):
                layers[i] = ReplacementBlock(hidden_size)
                layers_replaced += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "layers_removed": layers_replaced,
            "target_removed": self.remove_layers,
            "hidden_size": hidden_size,
        })
        return self.model, stats


@COMPRESSOR_REGISTRY.register("replaceme_factory")
def replaceme_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """ReplaceMe compressor factory function."""
    compressor = ReplaceMeCompressor(model, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model

