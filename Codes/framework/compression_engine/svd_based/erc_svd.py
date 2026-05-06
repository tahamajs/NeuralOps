"""ERC-SVD: Error-controlled SVD. Registered as 'erc'."""

import torch
import torch.nn as nn
from ..core.base import BaseCompressor
from ..core.registry import COMPRESSOR_REGISTRY

@COMPRESSOR_REGISTRY.register("erc")
class ERCCompressor(BaseCompressor):
    def __init__(self, model, allowed_error=0.01, **kwargs):
        super().__init__(model, name="erc", **kwargs)
        self.allowed_error = allowed_error

    def _compress_layer(self, W):
        U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
        energy = torch.cumsum(S**2, dim=0) / (S**2).sum()
        idx = torch.where(energy >= (1 - self.allowed_error))[0]
        rank = idx[0].item() + 1 if len(idx) > 0 else 1
        return U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = self._compress_layer(module.weight.data)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()