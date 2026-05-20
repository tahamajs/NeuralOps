"""
HashedNet Compressor: replaces nn.Linear with HashedLinear.
Registered as 'hashed_net'.
"""

import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY
from .hashed_linear import HashedLinear


class HashedNetCompressor(BaseCompressor):
    """
    Compress a model by replacing every nn.Linear layer with HashedLinear,
    which shares weights via a hash table.

    Args:
        model: PyTorch model.
        n_buckets: number of hash buckets (size of weight table).
    """

    def __init__(self, model, n_buckets=1024, **kwargs):
        super().__init__(model, name="hashed_net", **kwargs)
        self.n_buckets = n_buckets

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Replace with HashedLinear initialized from existing weights
                new_layer = HashedLinear.from_linear(module, self.n_buckets)
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
    




@COMPRESSOR_REGISTRY.register("hashed_net")
def hashed_net_factory(model, comp_config):
    n_buckets = getattr(comp_config, 'hashed_net_buckets', 1024)
    compressor = HashedNetCompressor(model, n_buckets=n_buckets)
    compressed_model, _ = compressor.compress()
    return compressed_model

