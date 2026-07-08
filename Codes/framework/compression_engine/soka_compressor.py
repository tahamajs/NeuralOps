"""
Low‑Rank Linear Compressor (formerly "SoKA").
Replaces nn.Linear with factorised U @ V, initialised from truncated SVD.
Registered as 'soka'.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class LowRankLinear(nn.Module):
    """Linear layer stored as W ≈ U @ V, implemented via two small matmuls."""

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
        # x: (..., in_features)
        return F.linear(F.linear(x, self.V), self.U.t()) + self.bias


class LowRankCompressor(BaseCompressor):
    """
    Compress all nn.Linear layers by replacing them with LowRankLinear.
    The rank can be fixed (soka_rank) or chosen per layer via an energy
    threshold (soka_energy).
    """

    def __init__(
        self, model: nn.Module, rank: int = 8, energy_threshold: float = None, **kwargs
    ):
        super().__init__(model, name="soka_compressor", **kwargs)
        self.rank = rank
        self.energy_threshold = energy_threshold
        self.compressed_layers = []

    def _select_rank(self, S: torch.Tensor) -> int:
        if self.energy_threshold is None:
            return min(self.rank, len(S))
        total = (S**2).sum()
        cum = torch.cumsum(S**2, dim=0) / total
        idx = torch.where(cum >= self.energy_threshold)[0]
        r = idx[0].item() + 1 if len(idx) > 0 else 1
        return min(r, len(S))

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        self._replace_all_linear()
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()

    def _replace_all_linear(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.float()
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                r = self._select_rank(S)

                new_layer = LowRankLinear(module.in_features, module.out_features, r)
                new_layer.U.data = (U[:, :r] * S[:r].unsqueeze(0)).float()
                new_layer.V.data = Vh[:r, :].float()
                if module.bias is not None:
                    new_layer.bias.data.copy_(module.bias.data)

                # Replace in model
                parent = self._get_parent(name)
                if parent is not None:
                    child_name = name.split(".")[-1]
                    setattr(parent, child_name, new_layer)

                self.compressed_layers.append(
                    {
                        "name": name,
                        "original_shape": module.weight.shape,
                        "rank": r,
                    }
                )

    def _get_parent(self, full_name: str):
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None
        return parent

    def get_compression_stats(self) -> Dict[str, Any]:
        stats = super().get_compression_stats()
        stats.update(
            {
                "total_layers": len(self.compressed_layers),
                "method": "low_rank_svd",
                "fixed_rank": self.rank,
            }
        )
        return stats


@COMPRESSOR_REGISTRY.register("soka")
def soka_factory(model, comp_config):
    rank = getattr(comp_config, "soka_rank", 8)
    energy = getattr(comp_config, "soka_energy", None)
    compressor = LowRankCompressor(model, rank=rank, energy_threshold=energy)
    compressed_model, _ = compressor.compress()
    return compressed_model
