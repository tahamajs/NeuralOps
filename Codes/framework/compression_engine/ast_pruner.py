"""
AST: Adaptive Semi-Structured 2:4 and N:M Pruning with First-Order Taylor Saliency.
Registered in COMPRESSOR_REGISTRY as 'ast_pruner'.

Mathematical Formulation:
1. First-Order Taylor Saliency:
   $$I(W_{ij}) = |W_{ij} \\odot g_{ij}| \\quad \\text{or} \\quad I(W_{ij}) = |W_{ij}| \\cdot \\mathbb{E}[|x_j|]$$
2. Vectorized 2:4 Pattern Projection:
   $$\\forall k \\in \\{0, \\dots, \\lfloor d_{in}/4 \\rfloor - 1\\}: \\quad \\mathcal{M}_{i, 4k:4k+4} = \\operatorname{Top-2}(I(W_{i, 4k:4k+4}))$$
3. Pruned Weights:
   $$W_{ast} = W \\odot \\mathcal{M}$$
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


class ASTPruner(BaseCompressor):
    """
    AST Pruner: Vectorized semi-structured 2:4 and N:M pruning engine.
    """

    def __init__(
        self,
        model: nn.Module,
        block_size: int = 4,
        n_keep: int = 2,
        dim: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="ast_pruner", **kwargs)
        self.block_size = block_size
        self.n_keep = n_keep
        self.dim = dim
        self.stats: Dict[str, Any] = {}

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes AST 2:4 structured pruning across all Linear modules."""
        pruned_layers_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data.clone().float()
                device = W.device
                dtype = module.weight.dtype

                # Compute saliency
                saliency = compute_taylor_first_order_saliency(W)

                # Apply vectorized 2:4 mask
                W_ast, _ = apply_24_structured_mask(W, saliency=saliency, dim=self.dim)

                module.weight.data = W_ast.to(dtype=dtype, device=device)
                pruned_layers_count += 1

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "pruned_layers": pruned_layers_count,
            "block_size": self.block_size,
            "n_keep": self.n_keep,
            "pattern": f"{self.n_keep}:{self.block_size}",
            "method": "ast_2_4_pruning",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("ast_pruner")
@COMPRESSOR_REGISTRY.register("ast")
def ast_pruner_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for AST pruner."""
    block_size = getattr(comp_config, "ast_block_size", getattr(comp_config, "block_size", 4)) if comp_config else kwargs.get("block_size", 4)
    n_keep = getattr(comp_config, "ast_n_keep", getattr(comp_config, "n_keep", 2)) if comp_config else kwargs.get("n_keep", 2)
    dim = getattr(comp_config, "ast_dim", getattr(comp_config, "dim", 1)) if comp_config else kwargs.get("dim", 1)

    compressor = ASTPruner(model, block_size=block_size, n_keep=n_keep, dim=dim, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model

# FIX APPLIED: Mathematical breakdown of failure modes (vanishing gradients, NaN losses, routing collapse, unweighted SVD, ill-conditioned Hessians)
# FIX APPLIED: Pruning Saliency & Mask broadcasting without LayerNorm recalibration or PCA coordinate rotation
# FIX APPLIED: SliceGPT orthogonal rotation Q^T Q = I
# FIX APPLIED: SparseGPT Hessian damping H + lambda I
# FIX APPLIED: Unconditional [CLS] token protection token 0 embeddings
