"""
Tensor Train Compressor: replaces eligible nn.Linear and Conv1D layers with TTLinearLayer.
Registered in COMPRESSOR_REGISTRY as "tt".
Production-hardened with:
- Relative Frobenius error validation
- Adaptive rank retries
- Conv1D support (GPT-2)
- Protection of critical classifier/head layers
- Shape verification and safe module replacement
- Detailed per-layer stats logging
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

from .tt_linear import TTLinearLayer

logger = logging.getLogger(__name__)


class TTCompressor(BaseCompressor):
    """
    Compressor that replaces nn.Linear/Conv1D layers with Tensor-Train (TTLinearLayer) modules.

    Args:
        model: PyTorch model to compress.
        rank: Rank parameter (alias for tt_rank).
        tt_rank: TT-rank bound for tensor-train cores.
        cfg: Optional configuration object.
        target_layers: Optional list of layer names or substrings to compress.
        max_error: Maximum allowed relative reconstruction error (default 0.20).
        max_retries: Number of rank-step retries on error threshold excess (default 3).
        fallback_to_original: If True, keep original layer on high error (default True).
        **kwargs: Additional arguments passed to BaseCompressor.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: Optional[int] = None,
        tt_rank: int = 16,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        max_error: float = 0.20,
        max_retries: int = 3,
        fallback_to_original: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="tt", **kwargs)
        comp_cfg = kwargs.get("comp_config", cfg)
        if comp_cfg is not None:
            rank = getattr(comp_cfg, "tt_rank", getattr(comp_cfg, "rank", rank))
            max_error = getattr(comp_cfg, "max_error", max_error)
            max_retries = getattr(comp_cfg, "max_retries", max_retries)
            fallback_to_original = getattr(comp_cfg, "fallback_to_original", fallback_to_original)

        if "max_error" in kwargs and kwargs["max_error"] is not None:
            max_error = float(kwargs["max_error"])
        if "max_retries" in kwargs and kwargs["max_retries"] is not None:
            max_retries = int(kwargs["max_retries"])
        if "fallback_to_original" in kwargs:
            fallback_to_original = bool(kwargs["fallback_to_original"])

        self.tt_rank = max(1, int(rank or tt_rank or 16))
        self.cfg = comp_cfg
        self.target_layers = target_layers or getattr(comp_cfg, "target_layers", None)
        self.max_error = float(max_error)
        self.max_retries = max(0, int(max_retries))
        self.fallback_to_original = bool(fallback_to_original)
        self.compressed_layers: List[Dict[str, Any]] = []

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        # Protect critical classification heads, poolers, lm_heads, and embeddings
        if any(k in name.lower() for k in ["classifier", "score", "lm_head", "pooler", "embedding"]):
            return False

        cls_name = module.__class__.__name__
        is_linear_like = (
            isinstance(module, nn.Linear)
            or "conv1d" in cls_name.lower()
            or (
                hasattr(module, "weight")
                and hasattr(module.weight, "ndim")
                and module.weight.ndim == 2
                and not isinstance(module, (nn.Embedding, nn.LayerNorm, nn.RMSNorm))
            )
        )
        if not is_linear_like:
            return False
        if isinstance(module, TTLinearLayer):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_layers and not any(t in name for t in self.target_layers):
            return False
        return True

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Replace linear/Conv1D layers with TTLinearLayer and return (model, stats)."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        total_error = 0.0

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            raw_w = module.weight.data
            cls_name = module.__class__.__name__
            is_conv1d = "conv1d" in cls_name.lower() or (
                raw_w.ndim == 2
                and hasattr(module, "in_features")
                and module.in_features == raw_w.shape[0]
                and raw_w.shape[0] < raw_w.shape[1]
            )

            if is_conv1d:
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            else:
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]

            try:
                tt_layer = TTLinearLayer.from_linear(
                    module,
                    tt_rank=self.tt_rank,
                    max_error=self.max_error,
                    max_retries=self.max_retries,
                )
            except Exception as e:
                logger.warning("TT-SVD decomposition failed for %s: %s. Keeping original module.", name, e)
                continue

            rel_err = getattr(tt_layer, "reconstruction_error", 0.0)
            chosen_rank = getattr(tt_layer, "tt_rank", self.tt_rank)

            if rel_err > self.max_error:
                logger.warning(
                    "TTCompressor: %s relative error (%.4f) exceeded threshold (%.4f).",
                    name,
                    rel_err,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("TTCompressor: Keeping original module for %s.", name)
                    continue

            # Verification of shape integrity
            if tt_layer.in_features != in_f or tt_layer.out_features != out_f:
                logger.error("TTCompressor: Shape mismatch for %s. Keeping original.", name)
                continue

            replace_module(self.model, name, tt_layer)
            self.compressed_layers.append(
                {
                    "name": name,
                    "original_shape": (in_f, out_f),
                    "tt_rank": chosen_rank,
                    "rel_error": float(rel_err),
                }
            )
            total_error += rel_err

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (total_error / len(self.compressed_layers)) if self.compressed_layers else 0.0
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "num_compressed_layers": len(self.compressed_layers),
                "tt_rank": self.tt_rank,
                "avg_reconstruction_error": avg_err,
                "max_error": self.max_error,
            }
        )
        logger.info(
            "TT compression finished: %d layers compressed, avg error: %.4f, param reduction: %.2f%%",
            len(self.compressed_layers),
            avg_err,
            stats.get("param_reduction_pct", 0.0),
        )
        return self.model, stats

    def compress_linear(self, linear_layer: nn.Module) -> TTLinearLayer:
        """Compress a single linear or Conv1D layer into TTLinearLayer."""
        return TTLinearLayer.from_linear(
            linear_layer,
            tt_rank=self.tt_rank,
            max_error=self.max_error,
            max_retries=self.max_retries,
        )


@COMPRESSOR_REGISTRY.register("tt")
def tt_compressor_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for Tensor Train compressor."""
    tt_rank = rank or getattr(comp_config, "tt_rank", None) or getattr(comp_config, "rank", 16)
    compressor = TTCompressor(model, tt_rank=tt_rank, cfg=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model

