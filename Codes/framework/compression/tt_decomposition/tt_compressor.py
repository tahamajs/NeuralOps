"""
Tensor Train Compressor: replaces all nn.Linear layers with TTLinearLayer.
Registered in COMPRESSOR_REGISTRY as "tt".
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from ...compression.core.registry import COMPRESSOR_REGISTRY
from ...compression.core.base import BaseCompressor
from .tt_linear import TTLinearLayer


@COMPRESSOR_REGISTRY.register("tt")
class TTCompressor(BaseCompressor):
    """
    Compress a model using Tensor Train (TT) decomposition on all linear layers.

    Args:
        model: PyTorch model to compress.
        tt_rank: rank of the TT decomposition (default 16).
        use_fallback: if True, uses SVD fallback even if torchtt installed.
    """

    def __init__(self, model: nn.Module, tt_rank: int = 16, use_fallback: bool = False, **kwargs):
        super().__init__(model, name="tt_compressor", **kwargs)
        self.tt_rank = tt_rank
        self.use_fallback = use_fallback
        self.compressed_layers = []

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        self._replace_all_linear()
        self._compressed_params = self._count_params(self.model)
        self._compression_stats = self._compute_stats()
        return self.model, self.get_compression_stats()

    def _replace_all_linear(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Replace with TTLinearLayer
                tt_layer = TTLinearLayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    tt_rank=self.tt_rank,
                    bias=module.bias is not None,
                )
                # If we have pretrained weights, transfer them
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

    def _compute_stats(self) -> Dict[str, Any]:
        total_original = self._original_params
        total_compressed = self._compressed_params
        return {
            'total_layers_compressed': len(self.compressed_layers),
            'compression_ratio': total_compressed / total_original if total_original > 0 else 1.0,
            'parameters_saved': total_original - total_compressed,
            'reduction_percentage': (1 - total_compressed / total_original) * 100 if total_original > 0 else 0,
            'compressed_layers': self.compressed_layers,
            'method': 'tensor_train',
            'rank': self.tt_rank,
        }