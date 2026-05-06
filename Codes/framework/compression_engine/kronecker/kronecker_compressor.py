"""
Kronecker Product Compressor: replaces nn.Linear with KroneckerLinear.
Registered as 'kronecker'.
"""

import torch.nn as nn
from ..core.base import BaseCompressor
from ..core.registry import COMPRESSOR_REGISTRY
from .kronecker_linear import KroneckerLinear


@COMPRESSOR_REGISTRY.register("kronecker")
class KroneckerCompressor(BaseCompressor):
    """
    Compresses a model by replacing every nn.Linear layer with a
    Kronecker product decomposition (A ⊗ B) when dimensions allow,
    falling back to a low‑rank SVD otherwise.

    Args:
        model: PyTorch model.
        **kwargs: passed to BaseCompressor.
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, name="kronecker", **kwargs)

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Replace with KroneckerLinear
                new_layer = KroneckerLinear.from_linear(module)
                parent = self._get_parent(name)
                if parent is not None:
                    child_name = name.split('.')[-1]
                    setattr(parent, child_name, new_layer)

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()

    def _get_parent(self, full_name):
        parts = full_name.split('.')
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None
        return parent