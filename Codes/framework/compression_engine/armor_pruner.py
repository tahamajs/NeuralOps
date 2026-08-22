"""
ARMOR: Adaptive Resilient Matrix Orthogonal Rank Pruning (2:4 Sparse Core + Low-Rank Correction).
Registered in COMPRESSOR_REGISTRY as 'armor_pruner'.

Mathematical Formulation:
1. Saliency Metric (First-Order Taylor or Second-Order Hessian):
   $$S_{ij} = |W_{ij} \\odot g_{ij}| \\quad \\text{or} \\quad S_{ij} = \\frac{|W_{ij}|^2}{2 [H^{-1}]_{jj}}$$
2. 2:4 Semi-Structured Pruning:
   $$W_{sparse} = \\Pi_{2:4}(W, S)$$
3. Pruning Residual:
   $$R = W - W_{sparse}$$
4. Truncated Singular Value Decomposition of Residual:
   $$R = U \\Sigma V^\\top \\approx U_r \\Sigma_r V_r^\\top = L_r$$
5. Error-Compensated Weight Representation:
   $$W_{armor} = W_{sparse} + L_r$$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import (
    apply_24_structured_mask,
    compute_taylor_first_order_saliency,
)

logger = logging.getLogger(__name__)


class ARMORPruner(BaseCompressor):
    """
    ARMOR Pruner: Combines 2:4 structured sparsity with low-rank error compensation.
    """

    def __init__(
        self,
        model: nn.Module,
        block_size: int = 4,
        rank: int = 8,
        sparsity: float = 0.5,
        structured_24: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="armor_pruner", **kwargs)
        self.block_size = block_size
        self.rank = max(1, int(rank))
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.structured_24 = structured_24
        self.stats: Dict[str, Any] = {}

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes ARMOR pruning with low-rank error correction."""
        pruned_layers_count = 0
        total_correction_energy = 0.0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.clone().float()
                device = W.device
                dtype = module.weight.dtype

                # 1. Compute saliency
                saliency = compute_taylor_first_order_saliency(W)

                # 2. Compute structured 2:4 or magnitude sparse baseline
                if self.structured_24:
                    W_sparse, _ = apply_24_structured_mask(W, saliency=saliency, dim=1)
                else:
                    flat = saliency.reshape(-1)
                    k = max(1, int(flat.numel() * (1.0 - self.sparsity)))
                    thr = torch.topk(flat, k, largest=True).values[-1]
                    mask = (saliency >= thr).float()
                    W_sparse = W * mask

                # 3. Residual Low-Rank Error Correction
                residual = W - W_sparse
                try:
                    U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
                    r = min(self.rank, len(S))
                    if r > 0:
                        correction = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vh[:r, :])
                        total_correction_energy += torch.norm(correction).item()
                        W_armor = W_sparse + correction
                    else:
                        W_armor = W_sparse
                except Exception as e:
                    logger.debug("SVD failed on residual in %s: %s", name, e)
                    W_armor = W_sparse

                module.weight.data = W_armor.to(dtype=dtype, device=device)
                pruned_layers_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "pruned_layers": pruned_layers_count,
            "rank": self.rank,
            "structured_24": self.structured_24,
            "total_correction_energy": total_correction_energy,
            "method": "armor_sparse_plus_lowrank",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("armor_pruner")
@COMPRESSOR_REGISTRY.register("armor")
def armor_pruner_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for ARMOR pruner."""
    block_size = getattr(comp_config, "armor_block_size", getattr(comp_config, "block_size", 4)) if comp_config else kwargs.get("block_size", 4)
    rank = getattr(comp_config, "armor_rank", getattr(comp_config, "rank", 8)) if comp_config else kwargs.get("rank", 8)
    structured_24 = getattr(comp_config, "armor_structured_24", getattr(comp_config, "structured_24", True)) if comp_config else kwargs.get("structured_24", True)

    compressor = ARMORPruner(
        model,
        block_size=block_size,
        rank=rank,
        structured_24=structured_24,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
