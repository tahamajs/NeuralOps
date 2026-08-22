"""
ProxSparse: Proximal Gradient Regularized Structured & Semi-Structured Sparsity.
Registered in COMPRESSOR_REGISTRY as 'proxsparse'.

Mathematical Formulation:
$$\\min_W \\frac{1}{2} \\|W - W_0\\|_F^2 + \\lambda \\|W\\|_1 + \\mathcal{I}_{\\mathcal{S}}(W)$$

Proximal Operators:
1. Soft-Thresholding:
   $$\\mathcal{S}_{\\lambda}(W) = \\operatorname{sign}(W) \\odot \\max(|W| - \\lambda, 0)$$
2. Sparsity Projection (2:4 Semi-Structured or Block/Percentile):
   $$\\Pi_{\\mathcal{S}}(W) = \\operatorname{Top-}(1-s)(W) \\quad \\text{or} \\quad \\Pi_{2:4}(W)$$
3. Iterative Proximal Gradient Descent:
   $$W^{(t+1)} = \\Pi_{\\mathcal{S}}\\left( W^{(t)} - \\eta (W^{(t)} - \\mathcal{S}_{\\lambda}(W_0)) \\right)$$
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


class ProxSparsePruner(BaseCompressor):
    """
    ProxSparse: Regularized learning of structured and semi-structured sparsity.
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        lambda_prox: float = 0.01,
        steps: int = 5,
        structured_24: bool = False,
        block_size: Tuple[int, int] = (4, 4),
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="proxsparse", **kwargs)
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.lambda_prox = lambda_prox
        self.steps = max(1, int(steps))
        self.structured_24 = structured_24
        self.block_size = block_size
        self.stats: Dict[str, Any] = {}

    def _soft_threshold(self, tensor: torch.Tensor, threshold: float) -> torch.Tensor:
        """Soft-thresholding operator prox_{\lambda ||.||_1}."""
        return torch.sign(tensor) * torch.clamp(torch.abs(tensor) - threshold, min=0.0)

    def _project_sparse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Projects continuous tensor onto sparse constraint set."""
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
        """Executes proximal gradient sparsity iterations across all Linear modules."""
        pruned_layers_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.clone().float()
                device = W.device
                dtype = module.weight.dtype

                W_curr = W.clone()
                for _ in range(self.steps):
                    # Soft-threshold step
                    W_soft = self._soft_threshold(W_curr, self.lambda_prox)
                    # Sparsity projection step
                    W_proj = self._project_sparse(W_soft)
                    # Proximal gradient update
                    W_curr = W_curr - 0.5 * (W_curr - W_proj)

                # Final projection
                W_final = self._project_sparse(W_curr)
                module.weight.data = W_final.to(dtype=dtype, device=device)
                pruned_layers_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "pruned_layers": pruned_layers_count,
            "sparsity_target": self.sparsity,
            "lambda_prox": self.lambda_prox,
            "structured_24": self.structured_24,
            "steps": self.steps,
            "method": "proxsparse_regularized",
        })
        self.stats = stats
        return self.model, stats


# Aliases
ProxSparseCompressor = ProxSparsePruner


@COMPRESSOR_REGISTRY.register("proxsparse")
def proxsparse_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for ProxSparse pruner."""
    sparsity = getattr(comp_config, "proxsparse_sparsity", getattr(comp_config, "sparsity", 0.5)) if comp_config else kwargs.get("sparsity", 0.5)
    lambda_prox = getattr(comp_config, "proxsparse_lambda", getattr(comp_config, "lambda_prox", 0.01)) if comp_config else kwargs.get("lambda_prox", 0.01)
    steps = getattr(comp_config, "proxsparse_steps", getattr(comp_config, "steps", 5)) if comp_config else kwargs.get("steps", 5)
    structured_24 = getattr(comp_config, "proxsparse_structured_24", getattr(comp_config, "structured_24", False)) if comp_config else kwargs.get("structured_24", False)

    compressor = ProxSparsePruner(
        model,
        sparsity=sparsity,
        lambda_prox=lambda_prox,
        steps=steps,
        structured_24=structured_24,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
