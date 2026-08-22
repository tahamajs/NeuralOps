"""
SoKA: Second-Order Knowledge-Aware Low-Rank Linear Compressor.
Registered in COMPRESSOR_REGISTRY as 'soka'.

Mathematical Formulation:
1. Weight Matrix Factorization via SVD:
   $$W = U \\Sigma V^\\top \\approx U_r \\Sigma_r V_r^\\top = A B^\\top$$
   where $A = U_r \\Sigma_r^{1/2} \\in \\mathbb{R}^{d_{out} \\times r}$ and $B = V_r \\Sigma_r^{1/2} \\in \\mathbb{R}^{d_{in} \\times r}$.
2. Forward Execution:
   $$y = (x B) A^\\top + b$$
3. Singular Energy Threshold Rank Selection:
   $$r = \\min \\left\\{ k \\;\\middle|\\; \\frac{\\sum_{i=1}^k \\sigma_i^2}{\\sum_{i=1}^{\\min(d_{in}, d_{out})} \\sigma_i^2} \\ge \\alpha \\right\\}$$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


class LowRankLinear(nn.Module):
    """
    Factorized Linear Layer executing as y = (x @ V) @ U^T + b.
    Reduces parameter storage from (d_in * d_out) to r * (d_in + d_out).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, min(in_features, out_features)))

        # V: [in_features, rank] -> maps x to bottleneck
        # U: [out_features, rank] -> maps bottleneck to output
        self.V = nn.Parameter(torch.empty((in_features, self.rank), device=device, dtype=dtype))
        self.U = nn.Parameter(torch.empty((out_features, self.rank), device=device, dtype=dtype))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        nn.init.normal_(self.V, std=0.02)
        nn.init.normal_(self.U, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]
        bottleneck = torch.matmul(x, self.V)  # [..., rank]
        out = torch.matmul(bottleneck, self.U.T)  # [..., out_features]
        if self.bias is not None:
            out = out + self.bias
        return out


class LowRankCompressor(BaseCompressor):
    """
    SoKA Low-Rank Linear Compressor.
    Replaces dense nn.Linear layers with factorized LowRankLinear layers.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 8,
        energy_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="soka", **kwargs)
        self.rank = max(1, int(rank))
        self.energy_threshold = energy_threshold
        self.compressed_layers: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {}

    def _select_rank(self, S: torch.Tensor) -> int:
        """Determines rank dynamically based on singular energy or fixed target."""
        if self.energy_threshold is None:
            return min(self.rank, len(S))
        total_energy = torch.sum(S ** 2)
        if total_energy <= 0:
            return min(self.rank, len(S))
        cum_energy = torch.cumsum(S ** 2, dim=0) / total_energy
        idx = torch.where(cum_energy >= self.energy_threshold)[0]
        r = idx[0].item() + 1 if len(idx) > 0 else len(S)
        return max(1, min(r, len(S)))

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Replaces all Linear modules with LowRankLinear."""
        linear_layers: List[Tuple[str, nn.Linear]] = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append((name, module))

        for name, module in linear_layers:
            W = module.weight.data.clone().float()
            device = module.weight.device
            dtype = module.weight.dtype
            d_out, d_in = W.shape

            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                r = self._select_rank(S)
                sqrt_S = torch.sqrt(S[:r])

                new_layer = LowRankLinear(
                    in_features=d_in,
                    out_features=d_out,
                    rank=r,
                    bias=(module.bias is not None),
                    device=device,
                    dtype=dtype,
                )

                # Assign factorized weights: W ≈ (U_r sqrt(S)) @ (sqrt(S) V_r^T)
                # U: [d_out, r], V: [d_in, r]
                new_layer.U.data = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
                new_layer.V.data = (Vh[:r, :].T * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)

                if module.bias is not None:
                    new_layer.bias.data.copy_(module.bias.data)

                replace_module(self.model, name, new_layer)
                self.compressed_layers.append({
                    "name": name,
                    "original_shape": (d_out, d_in),
                    "rank": r,
                })
            except Exception as e:
                logger.debug("SVD decomposition error in %s: %s", name, e)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "total_layers": len(self.compressed_layers),
            "method": "soka_low_rank_svd",
            "fixed_rank": self.rank,
            "energy_threshold": self.energy_threshold,
        })
        self.stats = stats
        return self.model, stats


# Aliases for framework naming compatibility
SoKACompressor = LowRankCompressor


@COMPRESSOR_REGISTRY.register("soka")
@COMPRESSOR_REGISTRY.register("soka_compressor")
def soka_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for SoKA compressor."""
    rank = getattr(comp_config, "soka_rank", getattr(comp_config, "rank", 8)) if comp_config else kwargs.get("rank", 8)
    energy = getattr(comp_config, "soka_energy", getattr(comp_config, "energy_threshold", None)) if comp_config else kwargs.get("energy_threshold", None)

    compressor = LowRankCompressor(model, rank=rank, energy_threshold=energy, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
