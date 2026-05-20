"""
SoKA (Kronecker product) Compressor: replaces nn.Linear with low‑rank SVD,
approximating the Kronecker decomposition.
Registered in COMPRESSOR_REGISTRY as "soka".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY



class SoKALinear(nn.Module):
    """
    Linear layer implemented via low‑rank SVD: W ≈ U @ V.
    This serves as a placeholder for a true Kronecker product decomposition.
    """
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.V = nn.Parameter(torch.empty(rank, in_features))
        nn.init.normal_(self.U, std=0.02)
        nn.init.normal_(self.V, std=0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return F.linear(F.linear(x, self.V), self.U.t()) + self.bias


class SoKACompressor(BaseCompressor):
    """
    Compress a model using SoKA (Kronecker) low‑rank approximation.

    Args:
        model: PyTorch model.
        rank: decomposition rank (default 8).
    """

    def __init__(self, model: nn.Module, rank: int = 8, **kwargs):
        super().__init__(model, name="soka_compressor", **kwargs)
        self.rank = rank
        self.compressed_layers = []

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        self._replace_all_linear()
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()

    def _replace_all_linear(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # SVD approximation
                W = module.weight.data.float()
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                r = min(self.rank, min(W.shape))
                new_layer = SoKALinear(module.in_features, module.out_features, r)
                new_layer.U.data = (U[:, :r] * S[:r].unsqueeze(0)).float()
                new_layer.V.data = Vh[:r, :].float()
                if module.bias is not None:
                    new_layer.bias.data.copy_(module.bias.data)

                # Replace in model
                parent = self._get_parent(name)
                if parent is not None:
                    child_name = name.split('.')[-1]
                    setattr(parent, child_name, new_layer)

                self.compressed_layers.append({
                    'name': name,
                    'original_size': module.weight.shape,
                    'rank': r,
                })

    def _get_parent(self, full_name: str):
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
            'method': 'soka_kronecker',
            'rank': self.rank,
        })
        return stats
    


@COMPRESSOR_REGISTRY.register("soka")
def soka_factory(model, comp_config):
    rank = getattr(comp_config, 'soka_rank', 8)
    compressor = SoKACompressor(model, rank=rank)
    compressed_model, _ = compressor.compress()
    return compressed_model
