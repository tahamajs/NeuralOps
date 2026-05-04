"""
Tensor Train Compressor: replaces all nn.Linear layers with TTLinearLayer.
Registered in COMPRESSOR_REGISTRY as "tt".
"""

import torch.nn as nn
from typing import Dict, Any, Tuple
from ..core.base import BaseCompressor
from ..core.registry import COMPRESSOR_REGISTRY
from .tt_linear import TTLinearLayer


@COMPRESSOR_REGISTRY.register("tt")
class TTCompressor(BaseCompressor):
    """
    Compress a model using Tensor Train (TT) decomposition on all linear layers.

    Args:
        model: PyTorch model to compress.
        tt_rank: rank of the TT decomposition (default 16).
    """

    def __init__(self, model: nn.Module, tt_rank: int = 16, **kwargs):
        super().__init__(model, name="tt_compressor", **kwargs)
        self.tt_rank = tt_rank
        self.compressed_layers = []

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        self._replace_all_linear()
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()

    def _replace_all_linear(self):
        """Replace all nn.Linear modules with TTLinearLayer."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Create TT layer from pretrained weights
                tt_layer = TTLinearLayer.from_linear(module, self.tt_rank)
                # Set the layer in the parent module
                parent = self._get_parent(name)
                if parent is not None:
                    child_name = name.split('.')[-1]
                    setattr(parent, child_name, tt_layer)
                self.compressed_layers.append({
                    'name': name,
                    'original_size': module.weight.shape,
                    'compressed_size': (tt_layer.out_features, tt_layer.in_features),
                    'rank': self.tt_rank,
                })

    def _get_parent(self, full_name: str):
        """Return the parent module for a given dotted name."""
        parts = full_name.split('.')
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None
        return parent

    def get_compression_stats(self) -> Dict[str, Any]:
        stats = super().get_compression_stats()
        stats.update({
            'total_layers': len(self.compressed_layers),
            'method': 'tensor_train',
            'tt_rank': self.tt_rank,
        })
        return stats