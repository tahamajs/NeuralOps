"""
Kronecker Product Compressor: replaces nn.Linear and Conv1D layers with KroneckerLinear.
Registered in COMPRESSOR_REGISTRY as 'kronecker'. Production hardened with
reconstruction validation, Conv1D support, adaptive SVD fallback, and critical layer protection.
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

from .kronecker_linear import KroneckerLinear

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("kronecker")
class KroneckerCompressor(BaseCompressor):
    """
    Compresses a model by replacing eligible nn.Linear and Conv1D layers with a
    Kronecker product decomposition (A (x) B), falling back to adaptive low-rank SVD
    when dimensions do not allow composite factorization or error exceeds max_error.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - Automatic SVD fallback with rank escalation if Kronecker factorization exceeds max_error.
    - Safe fallback to original module if even SVD fallback exceeds max_error.
    - Conv1D support (GPT-2 c_attn, c_proj) via transposed weight projection.
    - Unified critical-layer protection via should_compress_linear.
    - Parameter reduction verification ensuring factorized parameters < original weights.
    - Comprehensive per-layer compression statistics and logging.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: Optional[int] = None,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="kronecker", **kwargs)
        conf = cfg or kwargs.get("comp_config", None)

        if conf is not None:
            rank = getattr(conf, "kronecker_rank", getattr(conf, "rank", rank))
            max_error = getattr(conf, "kronecker_max_error", getattr(conf, "max_error", max_error))
            fallback_to_original = getattr(
                conf, "kronecker_fallback_to_original", getattr(conf, "fallback_to_original", fallback_to_original)
            )
            target_layers = getattr(conf, "target_layers", target_layers)

        self.rank = int(rank) if rank is not None and rank > 0 else None
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_layers = target_layers
        self.cfg = conf

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, Optional[int]] = {}
        self.layer_modes: Dict[str, str] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        cls_name = module.__class__.__name__.lower()
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
        if not (is_linear or is_conv1d):
            return False
        if isinstance(module, KroneckerLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_layers is not None:
            return any(t in name for t in self.target_layers)
        return True

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Compress the model by substituting Linear/Conv1D layers with KroneckerLinear.

        Returns:
            Tuple of (compressed_model, stats_dictionary).
        """
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        self.layer_modes = {}

        targets = [
            (name, mod)
            for name, mod in list(self.model.named_modules())
            if self._is_target_module(name, mod)
        ]

        if not targets:
            logger.warning("KroneckerCompressor: No compressible layers found.")
            return self.model, self.get_compression_stats()

        for name, module in targets:
            cls_name = module.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
            raw_w = module.weight.data

            if is_conv1d:
                W_target = raw_w.t().float()
            else:
                W_target = raw_w.float()

            W_norm = torch.norm(W_target, p="fro").item()

            try:
                new_layer = KroneckerLinear.from_linear(
                    module,
                    rank=self.rank,
                    max_error=self.max_error,
                )
                W_comp = new_layer.get_weight().float()
                err = float(torch.norm(W_target - W_comp, p="fro").item() / max(W_norm, 1e-12))
            except Exception as e:
                logger.warning("KroneckerLinear factorization failed on %s: %s", name, e)
                new_layer = None
                err = 1.0

            self.layer_errors[name] = err
            mode = "kronecker" if (new_layer is not None and getattr(new_layer, "can_decompose", False)) else "svd"
            self.layer_modes[name] = mode
            self.layer_ranks[name] = getattr(new_layer, "rank", None) if new_layer is not None else None

            if new_layer is None or err > self.max_error:
                logger.warning(
                    "Kronecker reconstruction error %.4f on %s exceeded threshold %.4f.",
                    err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping Kronecker compression for %s (fallback to original).", name)
                    self.fallback_layers.append(name)
                    continue

            replace_module(self.model, name, new_layer)
            self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "fallback_layers": self.fallback_layers,
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "layer_errors": self.layer_errors,
                "layer_ranks": self.layer_ranks,
                "layer_modes": self.layer_modes,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "avg_reconstruction_error": avg_err,
                "compressed_layer_count": len(self.compressed_layers),
            }
        )
        logger.info(
            "Kronecker compression complete: %d compressed, %d fallback. Avg error: %.4f.",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
        )
        return self.model, stats

    def compress_linear(self, linear: nn.Module) -> Tuple[KroneckerLinear, float]:
        """Compress a single linear or Conv1D layer into KroneckerLinear."""
        cls_name = linear.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(linear, "nf")
        raw_w = linear.weight.data
        if is_conv1d:
            W_target = raw_w.t().float()
        else:
            W_target = raw_w.float()

        kron_layer = KroneckerLinear.from_linear(
            linear,
            rank=self.rank,
            max_error=self.max_error,
        )
        W_comp = kron_layer.get_weight().float()
        W_norm = torch.norm(W_target, p="fro").item()
        err = float(torch.norm(W_target - W_comp, p="fro").item() / max(W_norm, 1e-12))
        return kron_layer, err


@COMPRESSOR_REGISTRY.register("kronecker_factory")
def kronecker_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Kronecker compressor factory function."""
    rank = kwargs.pop("rank", rank)
    max_error = kwargs.pop("max_error", getattr(comp_config, "kronecker_max_error", getattr(comp_config, "max_error", 0.20)))
    fallback = kwargs.pop("fallback_to_original", getattr(
        comp_config, "kronecker_fallback_to_original", getattr(comp_config, "fallback_to_original", True)
    ))
    compressor = KroneckerCompressor(
        model,
        rank=rank,
        max_error=max_error,
        fallback_to_original=fallback,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = ["KroneckerCompressor", "kronecker_factory"]
