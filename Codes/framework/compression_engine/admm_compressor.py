"""
ADMM Compressor: Alternating Direction Method of Multipliers for Joint Pruning and Quantization.
Registered in COMPRESSOR_REGISTRY as 'admm' and 'admm_compressor'. Production hardened with
reconstruction validation, Conv1D support, FP32 numerical stability, and safe fallback.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import (
    apply_24_structured_mask,
    apply_block_mask,
)
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("admm")
@COMPRESSOR_REGISTRY.register("admm_compressor")
class ADMMCompressor(BaseCompressor):
    """
    ADMM-based Joint Structured Pruning and Non-Uniform / Uniform Quantization Engine.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - Safe fallback to original module if optimization produces NaN/Inf or error exceeds threshold.
    - Robust Conv1D support (GPT-2 c_attn, c_proj) via weight transposition.
    - Unified critical-layer protection via should_compress_linear.
    - Strict FP32 computation for ADMM auxiliary and dual variables (W, Z, Q, U1, U2).
    - Shape verification ensuring reconstructed weights match original layer dimensions.
    - Per-layer stats and error reporting.
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        quant_bits: int = 4,
        rho1: float = 0.1,
        rho2: float = 0.1,
        max_iter: int = 8,
        structured_24: bool = True,
        block_size: Tuple[int, int] = (4, 4),
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="admm", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            sparsity = getattr(cfg, "admm_sparsity", getattr(cfg, "sparsity", sparsity))
            quant_bits = getattr(cfg, "admm_quant_bits", getattr(cfg, "quant_bits", quant_bits))
            rho1 = getattr(cfg, "admm_rho1", getattr(cfg, "rho1", rho1))
            rho2 = getattr(cfg, "admm_rho2", getattr(cfg, "rho2", rho2))
            max_iter = getattr(cfg, "admm_max_iter", getattr(cfg, "max_iter", max_iter))
            structured_24 = getattr(cfg, "admm_structured_24", getattr(cfg, "structured_24", structured_24))
            max_error = getattr(cfg, "admm_max_error", getattr(cfg, "max_error", max_error))
            fallback_to_original = getattr(
                cfg, "admm_fallback_to_original", getattr(cfg, "fallback_to_original", fallback_to_original)
            )

        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.quant_bits = int(quant_bits)
        self.rho1 = float(rho1)
        self.rho2 = float(rho2)
        self.max_iter = int(max_iter)
        self.structured_24 = bool(structured_24)
        self.block_size = block_size
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_modules = target_modules

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_sparsities: Dict[str, float] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        cls_name = module.__class__.__name__.lower()
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
        if not (is_linear or is_conv1d):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def _quantize_projection(self, tensor: torch.Tensor) -> torch.Tensor:
        """Projects continuous tensor onto b-bit uniform quantization grid in FP32."""
        if self.quant_bits >= 32:
            return tensor
        scale = 2.0 ** (self.quant_bits - 1) - 1.0
        max_val = torch.max(torch.abs(tensor)).clamp(min=1e-6)
        normalized = torch.clamp(tensor / max_val, -1.0, 1.0)
        quantized = torch.round(normalized * scale) / scale
        return quantized * max_val

    def _prune_projection(self, tensor: torch.Tensor) -> torch.Tensor:
        """Projects continuous tensor onto target sparse constraint set S."""
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
        """Executes ADMM joint optimization across all target linear/Conv1D modules."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_sparsities = {}

        targets = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not targets:
            logger.warning("ADMM: No compressible layers found.")
            return self.model, self.get_compression_stats()

        for name, module in targets:
            cls_name = module.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
            raw_w = module.weight.data
            orig_shape = raw_w.shape
            device = raw_w.device
            dtype = raw_w.dtype

            if is_conv1d:
                W0 = raw_w.t().float()  # [out_features, in_features]
            else:
                W0 = raw_w.float()      # [out_features, in_features]

            norm_w0 = torch.norm(W0, p="fro").item()

            # Initialize ADMM variables in FP32
            W = W0.clone()
            try:
                Z = self._prune_projection(W0)
                Q = self._quantize_projection(W0)
                U1 = torch.zeros_like(W0)
                U2 = torch.zeros_like(W0)

                for it in range(self.max_iter):
                    denom = 1.0 + self.rho1 + self.rho2
                    W = (W0 + self.rho1 * (Z - U1) + self.rho2 * (Q - U2)) / denom

                    Z = self._prune_projection(W + U1)
                    Q = self._quantize_projection(W + U2)

                    U1 = U1 + W - Z
                    U2 = U2 + W - Q

                    if (
                        torch.isnan(W).any()
                        or torch.isnan(Z).any()
                        or torch.isnan(Q).any()
                        or torch.isinf(W).any()
                        or torch.isinf(Z).any()
                        or torch.isinf(Q).any()
                    ):
                        logger.warning("ADMM: NaN/Inf detected at iter %d on %s", it, name)
                        break

                W_final = self._quantize_projection(Z)
            except Exception as e:
                logger.warning("ADMM optimization failed on %s: %s", name, e)
                W_final = None

            if (
                W_final is None
                or torch.isnan(W_final).any()
                or torch.isinf(W_final).any()
            ):
                final_err = 1.0
            else:
                if norm_w0 < 1e-12:
                    final_err = 0.0
                else:
                    final_err = float(torch.norm(W0 - W_final, p="fro").item() / norm_w0)

            self.layer_errors[name] = final_err
            self.layer_sparsities[name] = 0.5 if self.structured_24 else self.sparsity

            if W_final is None or final_err > self.max_error:
                logger.warning(
                    "ADMM reconstruction error %.4f on %s exceeded threshold %.4f.",
                    final_err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping ADMM compression for %s (fallback to original).", name)
                    self.fallback_layers.append(name)
                    continue

            # Assign compressed weight ensuring shape match
            if is_conv1d:
                res_w = W_final.t().to(dtype=dtype, device=device)
            else:
                res_w = W_final.to(dtype=dtype, device=device)

            assert res_w.shape == orig_shape, f"Shape mismatch on {name}: {res_w.shape} vs {orig_shape}"
            module.weight.data.copy_(res_w)
            self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "layer_errors": self.layer_errors,
            "layer_sparsities": self.layer_sparsities,
            "sparsity_target": self.sparsity,
            "quant_bits": self.quant_bits,
            "structured_24": self.structured_24,
            "admm_iterations": self.max_iter,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "avg_reconstruction_error": avg_err,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "method": "admm_joint_prune_quant",
        })
        logger.info(
            "ADMM compression complete: %d compressed, %d fallback. Avg error: %.4f.",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("admm")
@COMPRESSOR_REGISTRY.register("admm_compressor")
def admm_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for ADMM compressor."""
    sparsity = getattr(comp_config, "admm_sparsity", getattr(comp_config, "sparsity", 0.5)) if comp_config else kwargs.get("sparsity", 0.5)
    quant_bits = getattr(comp_config, "admm_quant_bits", getattr(comp_config, "quant_bits", 4)) if comp_config else kwargs.get("quant_bits", 4)
    structured_24 = getattr(comp_config, "admm_structured_24", getattr(comp_config, "structured_24", True)) if comp_config else kwargs.get("structured_24", True)
    max_error = getattr(comp_config, "admm_max_error", getattr(comp_config, "max_error", 0.20)) if comp_config else kwargs.get("max_error", 0.20)
    fallback = getattr(comp_config, "admm_fallback_to_original", getattr(comp_config, "fallback_to_original", True)) if comp_config else kwargs.get("fallback_to_original", True)

    compressor = ADMMCompressor(
        model,
        sparsity=sparsity,
        quant_bits=quant_bits,
        structured_24=structured_24,
        max_error=max_error,
        fallback_to_original=fallback,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = ["ADMMCompressor", "admm_factory"]
