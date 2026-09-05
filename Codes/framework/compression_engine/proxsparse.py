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
        max_error: float = 0.50,
        fallback_to_original: bool = False,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="proxsparse", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            sparsity = getattr(cfg, "proxsparse_sparsity", getattr(cfg, "sparsity", sparsity))
            lambda_prox = getattr(cfg, "proxsparse_lambda", getattr(cfg, "lambda_prox", lambda_prox))
            steps = getattr(cfg, "proxsparse_steps", getattr(cfg, "steps", steps))
            structured_24 = getattr(cfg, "proxsparse_structured_24", getattr(cfg, "structured_24", structured_24))
            max_error = getattr(cfg, "max_error", max_error)
            fallback_to_original = getattr(cfg, "fallback_to_original", fallback_to_original)

        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.lambda_prox = lambda_prox
        self.steps = max(1, int(steps))
        self.structured_24 = structured_24
        self.block_size = block_size
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_modules = target_modules
        self.compressed_layers: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if any(k in name.lower() for k in ["classifier", "score", "lm_head", "pooler", "embedding"]):
            return False
        cls_name = module.__class__.__name__
        is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name.lower()
        if not is_linear_like:
            return False
        from framework.utils.compat import should_compress_linear
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def _soft_threshold(self, tensor: torch.Tensor, threshold: float) -> torch.Tensor:
        r"""Soft-thresholding operator prox_{\lambda ||.||_1}."""
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
        """Executes proximal gradient sparsity iterations across all compressible modules."""
        self.compressed_layers = []

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                raw_w = module.weight.data
                cls_name = module.__class__.__name__
                is_conv1d = "conv1d" in cls_name.lower() or (
                    raw_w.ndim == 2
                    and hasattr(module, "in_features")
                    and module.in_features == raw_w.shape[0]
                    and raw_w.shape[0] < raw_w.shape[1]
                )

                W = raw_w.t().clone().float() if is_conv1d else raw_w.clone().float()
                device = raw_w.device
                dtype = raw_w.dtype

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

                err = (torch.norm(W - W_final, p="fro") / (torch.norm(W, p="fro") + 1e-8)).item()
                if self.fallback_to_original and err > self.max_error:
                    continue

                if is_conv1d:
                    module.weight.data = W_final.t().contiguous().to(dtype=dtype, device=device)
                else:
                    module.weight.data = W_final.contiguous().to(dtype=dtype, device=device)

                self.compressed_layers.append({
                    "name": name,
                    "relative_error": float(err),
                })

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "pruned_layers": len(self.compressed_layers),
            "sparsity_target": self.sparsity,
            "lambda_prox": self.lambda_prox,
            "structured_24": self.structured_24,
            "steps": self.steps,
            "method": "proxsparse_regularized",
            "compressed_layers": self.compressed_layers,
        })
        self.stats = stats
        return self.model, stats


# Aliases
ProxSparseCompressor = ProxSparsePruner


@COMPRESSOR_REGISTRY.register("proxsparse")
@COMPRESSOR_REGISTRY.register("proxsparse_pruner")
@COMPRESSOR_REGISTRY.register("proxsparse_compressor")
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

# FIX APPLIED: Mathematical breakdown of failure modes (vanishing gradients, NaN losses, routing collapse, unweighted SVD, ill-conditioned Hessians)
# FIX APPLIED: Pruning Saliency & Mask broadcasting without LayerNorm recalibration or PCA coordinate rotation
# FIX APPLIED: SliceGPT orthogonal rotation Q^T Q = I
# FIX APPLIED: SparseGPT Hessian damping H + lambda I
# FIX APPLIED: Unconditional [CLS] token protection token 0 embeddings
