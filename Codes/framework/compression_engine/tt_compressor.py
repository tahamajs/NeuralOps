"""
Tensor Train Compressor: replaces all nn.Linear layers with TTLinearLayer.
Registered in COMPRESSOR_REGISTRY as "tt".
"""

import torch.nn as nn
from typing import Dict, Any, Tuple
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY
from .tt_linear import TTLinearLayer


class TTCompressor(BaseCompressor):
    """
    Internal compressor class that applies TT decomposition to all linear layers.
    Not registered directly – used by the factory function below.
    """
    def __init__(self, model: nn.Module, tt_rank: int = 16, **kwargs):
        super().__init__(model, name="tt_compressor", **kwargs)
        self.tt_rank = tt_rank
        self.compressed_layers = []

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Replace linear layers and return compressed model + stats."""
        self._replace_all_linear()
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()

    def _replace_all_linear(self):
        """Replace all nn.Linear modules with TTLinearLayer."""
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and not isinstance(module, TTLinearLayer):
                tt_layer = TTLinearLayer.from_linear(module, self.tt_rank)
                parent = self._get_parent_module(self.model, name)
                if parent is not None:
                    child_name = name.split('.')[-1]
                    setattr(parent, child_name, tt_layer)
                self.compressed_layers.append({
                    'name': name,
                    'original_shape': (module.in_features, module.out_features),
                    'compressed_rank': self.tt_rank,
                })

    def _count_params(self, module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def get_compression_stats(self) -> Dict[str, Any]:
        stats = super().get_compression_stats()
        stats.update({
            'compressed_layers': len(self.compressed_layers),
            'method': 'tensor_train',
            'tt_rank': self.tt_rank,
        })
        return stats


# ----------------------------------------------------------------------
# Registry entry point (factory function) – this is what the framework calls
# ----------------------------------------------------------------------
@COMPRESSOR_REGISTRY.register("tt")
def tt_compressor_factory(model: nn.Module, comp_config) -> nn.Module:
    """
    Factory function called by the framework.
    Expects comp_config to have attribute 'tt_rank' (default 16).
    Returns the compressed model (nn.Module) ready for training.
    """
    # Extract tt_rank from comp_config if available, else default 16
    tt_rank = getattr(comp_config, 'tt_rank', 16)
    compressor = TTCompressor(model, tt_rank=tt_rank)
    compressed_model, _ = compressor.compress()
    return compressed_model   # <-- returns nn.Module, not a compressor object
