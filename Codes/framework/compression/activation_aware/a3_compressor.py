"""A3: grouped QK/OV/MLP compression. Registered as 'a3'."""

import torch
import torch.nn as nn
from .core.base import BaseCompressor
from .core.registry import COMPRESSOR_REGISTRY

@COMPRESSOR_REGISTRY.register("a3")
class A3Compressor(BaseCompressor):
    def __init__(self, model, rank=64, **kwargs):
        super().__init__(model, name="a3", **kwargs)
        self.rank = rank

    def compress(self):
        for name, module in self.model.named_modules():
            # Example: compress all Linear layers in attention blocks to rank
            if isinstance(module, nn.Linear) and ('attention' in name.lower() or 'attn' in name.lower()):
                W = module.weight.data.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = min(self.rank, min(W.shape), len(S))
                W_approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
                module.weight.data = W_approx.to(dtype=module.weight.dtype)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()