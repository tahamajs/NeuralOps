"""
ADMM Pruner: Alternating Direction Method of Multipliers for Structured & Semi-Structured Pruning.
Registered in COMPRESSOR_REGISTRY as 'admm_pruner'.

Mathematical Formulation:
$$\\min_W \\frac{1}{2} \\|W - W_0\\|_F^2 + g(Z) \\quad \\text{s.t.} \\quad W = Z$$

ADMM Updates:
1. $Z^{(t+1)} = \\Pi_{\\mathcal{S}}(W^{(t)} + U^{(t)})$
2. $W^{(t+1)} = \\frac{1}{1 + \\rho} \\left( W_0 + \\rho (Z^{(t+1)} - U^{(t)}) \\right)$
3. $U^{(t+1)} = U^{(t)} + W^{(t+1)} - Z^{(t+1)}$
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
    apply_block_mask,
)

logger = logging.getLogger(__name__)


class ADMMPruner(BaseCompressor):
    """
    ADMM Pruner supporting unstructured sparsity, block-wise structured sparsity,
    and 2:4 semi-structured tensor core acceleration.
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        rho: float = 0.1,
        max_iter: int = 6,
        structured_24: bool = True,
        block_size: Tuple[int, int] = (4, 4),
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="admm_pruner", **kwargs)
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.rho = rho
        self.max_iter = max_iter
        self.structured_24 = structured_24
        self.block_size = block_size
        self.stats: Dict[str, Any] = {}

    def _project_sparse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Projects continuous weights onto sparse constraint set."""
        if self.structured_24:
            pruned, _ = apply_24_structured_mask(tensor, saliency=torch.abs(tensor), dim=1)
            return pruned
        elif self.block_size != (1, 1):
            pruned, _ = apply_block_mask(tensor, block_size=self.block_size, sparsity=self.sparsity)
            return pruned
        else:
            flat = tensor.abs().reshape(-1)
            k = max(1, int(flat.numel() * (1.0 - self.sparsity)))
            thr = torch.topk(flat, k, largest=True).values[-1]
            mask = (tensor.abs() >= thr).float()
            return tensor * mask

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes ADMM pruning across all Linear modules."""
        pruned_layers_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W0 = module.weight.data.clone().float()
                device = W0.device
                dtype = module.weight.dtype

                W = W0.clone()
                Z = self._project_sparse(W0)
                U = torch.zeros_like(W0)

                for it in range(self.max_iter):
                    # 1. Update Z (Sparse projection)
                    Z = self._project_sparse(W + U)

                    # 2. Update W
                    W = (W0 + self.rho * (Z - U)) / (1.0 + self.rho)

                    # 3. Update Dual variable U
                    U = U + W - Z

                # Final assignment to projection Z
                module.weight.data = Z.to(dtype=dtype, device=device)
                pruned_layers_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "pruned_layers": pruned_layers_count,
            "sparsity_target": self.sparsity,
            "structured_24": self.structured_24,
            "admm_iterations": self.max_iter,
            "method": "admm_pruner",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("admm_pruner")
def admm_pruner_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for ADMM pruner."""
    sparsity = getattr(comp_config, "admm_pruner_sparsity", getattr(comp_config, "sparsity", 0.5)) if comp_config else kwargs.get("sparsity", 0.5)
    rho = getattr(comp_config, "admm_pruner_rho", getattr(comp_config, "rho", 0.1)) if comp_config else kwargs.get("rho", 0.1)
    max_iter = getattr(comp_config, "admm_pruner_max_iter", getattr(comp_config, "max_iter", 6)) if comp_config else kwargs.get("max_iter", 6)
    structured_24 = getattr(comp_config, "admm_pruner_structured_24", getattr(comp_config, "structured_24", True)) if comp_config else kwargs.get("structured_24", True)

    compressor = ADMMPruner(
        model,
        sparsity=sparsity,
        rho=rho,
        max_iter=max_iter,
        structured_24=structured_24,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
