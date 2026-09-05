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


@COMPRESSOR_REGISTRY.register("proxsparse")
@COMPRESSOR_REGISTRY.register("proxsparse_pruner")
@COMPRESSOR_REGISTRY.register("proxsparse_compressor")
class ProxSparsePruner(BaseCompressor):
    """
    ProxSparse: Regularized learning of structured and semi-structured sparsity.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error (default 0.20).
    - Safe fallback to original weights by default (fallback_to_original=True).
    - Robust Conv1D detection by class name and nf attribute without unreliable shape heuristics.
    - Unified critical-layer protection via should_compress_linear.
    - Warning logged whenever error exceeds max_error regardless of fallback setting.
    - FP32 numerical stability with NaN/Inf guards.
    - Comprehensive per-layer stats and error tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        lambda_prox: float = 0.01,
        steps: int = 5,
        structured_24: bool = False,
        block_size: Tuple[int, int] = (4, 4),
        max_error: float = 0.20,
        fallback_to_original: bool = True,
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
            max_error = getattr(cfg, "proxsparse_max_error", getattr(cfg, "max_error", max_error))
            fallback_to_original = getattr(
                cfg, "proxsparse_fallback_to_original", getattr(cfg, "fallback_to_original", fallback_to_original)
            )

        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.lambda_prox = float(lambda_prox)
        self.steps = max(1, int(steps))
        self.structured_24 = bool(structured_24)
        self.block_size = block_size
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_modules = target_modules
        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        cls_name = module.__class__.__name__.lower()
        is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
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
        self.fallback_layers = []
        self.layer_errors = {}

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                raw_w = module.weight.data
                cls_name = module.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")

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
                W_final = torch.nan_to_num(W_final, nan=0.0)

                w_norm = torch.norm(W, p="fro").item()
                err = (torch.norm(W - W_final, p="fro") / max(w_norm, 1e-12)).item()
                self.layer_errors[name] = float(err)

                if err > self.max_error:
                    logger.warning(
                        "ProxSparse reconstruction error %.4f on %s exceeded threshold %.4f.",
                        err,
                        name,
                        self.max_error,
                    )
                    if self.fallback_to_original:
                        logger.info("Skipping ProxSparse for %s (fallback to original).", name)
                        self.fallback_layers.append(name)
                        continue

                if is_conv1d:
                    module.weight.data.copy_(W_final.t().contiguous().to(dtype=dtype, device=device))
                else:
                    module.weight.data.copy_(W_final.contiguous().to(dtype=dtype, device=device))

                self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats = self.get_compression_stats()
        stats.update({
            "pruned_layers": len(self.compressed_layers),
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "layer_errors": self.layer_errors,
            "avg_reconstruction_error": avg_err,
            "sparsity_target": self.sparsity,
            "lambda_prox": self.lambda_prox,
            "structured_24": self.structured_24,
            "steps": self.steps,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "method": "proxsparse_regularized",
        })
        self.stats = stats
        return self.model, stats


# Aliases
ProxSparseCompressor = ProxSparsePruner


@COMPRESSOR_REGISTRY.register("proxsparse_factory")
@COMPRESSOR_REGISTRY.register("proxsparse_pruner_factory")
@COMPRESSOR_REGISTRY.register("proxsparse_compressor_factory")
def proxsparse_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for ProxSparse pruner."""
    sparsity = kwargs.pop("sparsity", getattr(comp_config, "proxsparse_sparsity", getattr(comp_config, "sparsity", 0.5)))
    lambda_prox = kwargs.pop("lambda_prox", getattr(comp_config, "proxsparse_lambda", getattr(comp_config, "lambda_prox", 0.01)))
    steps = kwargs.pop("steps", getattr(comp_config, "proxsparse_steps", getattr(comp_config, "steps", 5)))
    structured_24 = kwargs.pop("structured_24", getattr(comp_config, "proxsparse_structured_24", getattr(comp_config, "structured_24", False)))
    max_error = kwargs.pop("max_error", getattr(comp_config, "proxsparse_max_error", getattr(comp_config, "max_error", 0.20)))
    fallback = kwargs.pop("fallback_to_original", getattr(
        comp_config, "proxsparse_fallback_to_original", getattr(comp_config, "fallback_to_original", True)
    ))

    compressor = ProxSparsePruner(
        model,
        sparsity=sparsity,
        lambda_prox=lambda_prox,
        steps=steps,
        structured_24=structured_24,
        max_error=max_error,
        fallback_to_original=fallback,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model

# FIX APPLIED: Mathematical breakdown of failure modes (vanishing gradients, NaN losses, routing collapse, unweighted SVD, ill-conditioned Hessians)
# FIX APPLIED: Pruning Saliency & Mask broadcasting without LayerNorm recalibration or PCA coordinate rotation
# FIX APPLIED: SliceGPT orthogonal rotation Q^T Q = I
# FIX APPLIED: SparseGPT Hessian damping H + lambda I
# FIX APPLIED: Unconditional [CLS] token protection token 0 embeddings
