"""ReplaceMe: removes last N encoder layers with a simple linear replacement. Registered as 'replaceme'."""

import torch.nn as nn
from framework.core.base import BaseCompressor


class ReplaceMeCompressor(BaseCompressor):
    def __init__(self, model, remove_layers=2, **kwargs):
        super().__init__(model, name="replaceme", **kwargs)
        self.remove_layers = remove_layers

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        if hasattr(self.model, "encoder"):
            layers = self.model.encoder.layer
            dim = self.model.config.hidden_size
            for i in range(len(layers) - self.remove_layers, len(layers)):
                replacement = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
                self.model.encoder.layer[i] = replacement
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()
