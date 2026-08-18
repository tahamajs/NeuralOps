from typing import Any, Dict, Tuple

import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


# FIX: Register ReplaceMeCompressor in COMPRESSOR_REGISTRY under canonical key 'replaceme'
@COMPRESSOR_REGISTRY.register("replaceme")
class ReplaceMeCompressor(BaseCompressor):
    def __init__(self, model: nn.Module, remove_layers: int = 2, **kwargs: Any) -> None:
        super().__init__(model, name="replaceme", **kwargs)
        self.remove_layers = remove_layers

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress the target module using this strategy."""
        if hasattr(self.model, "encoder"):
            layers = self.model.encoder.layer
            dim = getattr(self.model, "config", None)
            hidden_dim = getattr(dim, "hidden_size", 768) if dim is not None else 768
            for i in range(len(layers) - self.remove_layers, len(layers)):
                replacement = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
                self.model.encoder.layer[i] = replacement
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()
