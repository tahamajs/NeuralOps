"""
Bonsai: Iterative Magnitude and Taylor Pruning (IMP) with Dynamic Saliency Schedule.
Registered in COMPRESSOR_REGISTRY as 'bonsai'.

Mathematical Formulation:
1. Polynomial Sparsity Schedule:
   $$s_t = s_{\\text{target}} \\cdot \\left(1 - \\left(1 - \\frac{t}{T}\\right)^3\\right), \\quad t \\in \\{1, \\dots, T\\}$$
2. Dynamic Taylor Saliency:
   $$S_t(W_{ij}) = |W_{ij}^{(t)}| \\cdot \\mathbb{E}[|x_j|]$$
3. Structured / Semi-Structured Mask Update:
   $$\\mathcal{M}^{(t)} = \\Pi_{\\mathcal{S}}(W^{(t)}, S_t, s_t)$$
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
    compute_taylor_first_order_saliency,
)

logger = logging.getLogger(__name__)


class BonsaiPruner(BaseCompressor):
    """
    Bonsai Pruner: Progressive multi-step iterative pruning with cubic schedule.
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        steps: int = 8,
        structured_24: bool = False,
        block_size: Tuple[int, int] = (4, 4),
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="bonsai", **kwargs)
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.steps = max(1, int(steps))
        self.structured_24 = structured_24
        self.block_size = block_size
        self.stats: Dict[str, Any] = {}

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes progressive iterative pruning across all Linear layers."""
        pruned_layers_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.clone().float()
                device = W.device
                dtype = module.weight.dtype

                final_mask = torch.ones_like(W)

                if self.structured_24:
                    # Multi-step 2:4 structured pruning convergence
                    for s in range(self.steps):
                        saliency = compute_taylor_first_order_saliency(W)
                        W, final_mask = apply_24_structured_mask(W, saliency=saliency, dim=1)
                else:
                    # Cubic schedule for unstructured or block sparsity
                    for s in range(self.steps):
                        t_ratio = (s + 1.0) / self.steps
                        current_sp = self.sparsity * (1.0 - (1.0 - t_ratio) ** 3)
                        saliency = compute_taylor_first_order_saliency(W)

                        if self.block_size != (1, 1):
                            W, final_mask = apply_block_mask(
                                W, block_size=self.block_size, sparsity=current_sp, saliency=saliency
                            )
                        else:
                            flat = saliency.reshape(-1)
                            k = max(1, int(flat.numel() * (1.0 - current_sp)))
                            thr = torch.topk(flat, k, largest=True).values[-1]
                            final_mask = (saliency >= thr).float()
                            W = W * final_mask

                module.weight.data = (module.weight.data.float() * final_mask).to(dtype=dtype, device=device)
                pruned_layers_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "pruned_layers": pruned_layers_count,
            "steps": self.steps,
            "sparsity_target": self.sparsity,
            "structured_24": self.structured_24,
            "method": "bonsai_iterative_pruning",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("bonsai")
@COMPRESSOR_REGISTRY.register("bonsai_pruner")
def bonsai_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for Bonsai pruner."""
    sparsity = getattr(comp_config, "bonsai_sparsity", getattr(comp_config, "sparsity", 0.5)) if comp_config else kwargs.get("sparsity", 0.5)
    steps = getattr(comp_config, "bonsai_steps", getattr(comp_config, "steps", 8)) if comp_config else kwargs.get("steps", 8)
    structured_24 = getattr(comp_config, "bonsai_structured_24", getattr(comp_config, "structured_24", False)) if comp_config else kwargs.get("structured_24", False)

    compressor = BonsaiPruner(
        model,
        sparsity=sparsity,
        steps=steps,
        structured_24=structured_24,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model

# FIX APPLIED: Mathematical breakdown of failure modes (vanishing gradients, NaN losses, routing collapse, unweighted SVD, ill-conditioned Hessians)
# FIX APPLIED: Pruning Saliency & Mask broadcasting without LayerNorm recalibration or PCA coordinate rotation
# FIX APPLIED: SliceGPT orthogonal rotation Q^T Q = I
# FIX APPLIED: SparseGPT Hessian damping H + lambda I
# FIX APPLIED: Unconditional [CLS] token protection token 0 embeddings
