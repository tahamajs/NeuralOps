"""SALR: Unifies low-rank and sparse pruning. Registered as 'salr'."""

import torch
import torch.nn as nn
from ..core.base import BaseCompressor
from ..core.registry import COMPRESSOR_REGISTRY

class SALRLayer(nn.Module):
    """Low-rank + sparse mask layer."""
    def __init__(self, linear: nn.Linear, rank: int, sparsity: float):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.sparsity = sparsity
        U, S, Vt = torch.linalg.svd(linear.weight.data.float(), full_matrices=False)
        r = min(rank, min(linear.out_features, linear.in_features))
        self.U = nn.Parameter(U[:, :r].float())
        self.S = nn.Parameter(S[:r].float())
        self.Vt = nn.Parameter(Vt[:r, :].float())
        self.bias = nn.Parameter(linear.bias.data.clone() if linear.bias is not None else torch.zeros(linear.out_features))
        # sparse mask: randomly keep a fraction of elements in the core
        mask = (torch.rand(r, r) < (1 - sparsity)).float()
        self.register_buffer('mask', mask)

    def forward(self, x):
        S_sparse = self.S * self.mask[:self.rank, :self.rank]
        W = self.U @ torch.diag(S_sparse) @ self.Vt
        return x @ W.T + self.bias


@COMPRESSOR_REGISTRY.register("salr")
class SALRCompressor(BaseCompressor):
    def __init__(self, model, rank=64, sparsity=0.5, **kwargs):
        super().__init__(model, name="salr", **kwargs)
        self.rank = rank
        self.sparsity = sparsity
        self.compressed_layers = []

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                new_layer = SALRLayer(module, self.rank, self.sparsity)
                parent = self._get_parent(name)
                if parent is not None:
                    setattr(parent, name.split('.')[-1], new_layer)
                self.compressed_layers.append(name)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()

    def _get_parent(self, full_name):
        parts = full_name.split('.')
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
        return parent