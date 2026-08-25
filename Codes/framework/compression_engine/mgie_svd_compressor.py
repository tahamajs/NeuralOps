"""
MGIE-SVD with total rank budget: global entropy-proportional rank allocation.
Registered as 'mgie_budget'.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("mgie_budget")
class MGIESVDBudgetCompressor(BaseCompressor):
    """
    MGIE-SVD Budget Compressor: Global Entropy-Proportional Rank Allocation.

    Calculates Shannon singular entropy for all linear layers across the network,
    distributes a global total rank budget proportionally according to spectral information density,
    factorizes weights using symmetric square-root singular value splitting, and replaces
    dense layers with structural SVDLinear modules.

    Args:
        model: PyTorch model.
        total_rank_budget: Total sum of ranks across all compressed linear layers.
        rank: Optional per-layer fallback/override rank.
        target_modules: Optional list of layer name filters.
    """

    def __init__(
        self,
        model: nn.Module,
        total_rank_budget: int = 1000,
        rank: Optional[int] = None,
        target_modules: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="mgie_budget", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            total_rank_budget = getattr(
                cfg, "total_rank_budget", getattr(cfg, "rank_budget", total_rank_budget)
            )
            rank = getattr(cfg, "mgie_rank", getattr(cfg, "rank", rank))

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]

        self.total_rank_budget = int(total_rank_budget)
        self.rank = rank
        self.target_modules = target_modules

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if isinstance(module, SVDLinear):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    @staticmethod
    def _compute_entropy(W: torch.Tensor) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute singular entropy on squared singular values (energy distribution),
        normalized by layer parameter count sqrt(d_in * d_out), returning SVD components in FP32."""
        d_out, d_in = W.shape
        U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
        energy = S ** 2
        total_energy = energy.sum()
        if total_energy <= 1e-12:
            return 0.0, U, S, Vt
        p = torch.clamp(energy / total_energy, min=1e-12, max=1.0)
        entropy = -torch.sum(p * torch.log(p)).item()
        norm_factor = math.sqrt(float(d_in * d_out))
        normalized_entropy = entropy / max(1e-6, norm_factor)
        return max(0.0, float(normalized_entropy)), U, S, Vt

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress all linear layers according to global entropy budget."""
        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("MGIE-Budget: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        # Step 1: Collect singular entropies and SVD factors
        layer_data = {}
        total_entropy = 0.0
        for name, module in target_layers:
            ent, U, S, Vt = self._compute_entropy(module.weight.data)
            layer_data[name] = (module, ent, U, S, Vt)
            total_entropy += max(ent, 1e-4)

        # Step 2: Allocate rank budgets
        allocated_ranks = {}
        for name, (module, ent, U, S, Vt) in layer_data.items():
            max_r = min(module.in_features, module.out_features)
            if self.rank is not None:
                r = min(self.rank, max_r)
            else:
                frac = max(ent, 1e-4) / total_entropy
                r = int(round(frac * self.total_rank_budget))
            r = max(1, min(r, max_r))
            allocated_ranks[name] = r

            # Factorize with symmetric square-root singular values
            device = module.weight.device
            dtype = module.weight.dtype

            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
            V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(dtype=dtype, device=device)

            svd_layer = SVDLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=r,
                bias=module.bias is not None,
                dtype=dtype,
                device=device,
            )
            svd_layer.set_low_rank_factors(U_r, V_r)
            if module.bias is not None:
                svd_layer.bias.data.copy_(module.bias.data)

            replace_module(self.model, name, svd_layer)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats["allocated_ranks"] = allocated_ranks
        return self.model, stats


def mgie_budget_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """MGIE budget factory function."""
    compressor = MGIESVDBudgetCompressor(model, comp_config=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model


mgie_svd_budget_factory = mgie_budget_factory
mgie_svd_compressor_factory = mgie_budget_factory

__all__ = [
    "MGIESVDBudgetCompressor",
    "mgie_budget_factory",
    "mgie_svd_budget_factory",
    "mgie_svd_compressor_factory",
]
