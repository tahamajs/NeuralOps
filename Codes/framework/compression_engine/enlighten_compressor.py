"""
ENLighten: Low-Rank SVD + 2:4 Structured Sparsity (Photonic-Aligned).
Registered as 'enlighten'. Production hardened for multi-architecture compatibility
(DistilBERT, RoBERTa, GPT-2, Qwen3.5) with vectorization, error validation, and safe fallback.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def vectorized_24_prune(residual: torch.Tensor) -> torch.Tensor:
    """
    Vectorized 2:4 structured sparsity along row blocks (groups of 4 input columns).
    Handles arbitrary in_features by zero-padding to a multiple of 4.
    """
    out_f, in_f = residual.shape
    rem = in_f % 4
    if rem != 0:
        pad = 4 - rem
        res_padded = torch.nn.functional.pad(residual, (0, pad), mode="constant", value=0.0)
    else:
        pad = 0
        res_padded = residual

    num_blocks = res_padded.shape[1] // 4
    # Reshape to [out_features, num_blocks, 4]
    blocks = res_padded.view(out_f, num_blocks, 4)
    _, top_idx = torch.topk(blocks.abs(), k=2, dim=-1)
    mask = torch.zeros_like(blocks)
    mask.scatter_(-1, top_idx, 1.0)
    pruned_blocks = blocks * mask
    pruned_padded = pruned_blocks.view(out_f, -1)

    if pad > 0:
        return pruned_padded[:, :in_f]
    return pruned_padded


def unstructured_prune(residual: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Unstructured magnitude pruning based on target sparsity fraction."""
    if sparsity <= 0.0:
        return residual
    if sparsity >= 1.0:
        return torch.zeros_like(residual)
    k_prune = int(residual.numel() * sparsity)
    if k_prune <= 0:
        return residual
    flat = residual.abs().flatten()
    threshold = torch.kthvalue(flat, k_prune).values
    mask = (residual.abs() > threshold).to(residual.dtype)
    return residual * mask


@COMPRESSOR_REGISTRY.register("enlighten")
class ENLightenCompressor(BaseCompressor):
    """
    ENLighten Compressor: Low-Rank SVD + 2:4 Structured Sparsity.
    Decomposes weights into low-rank factors plus 2:4 structured sparse residuals,
    preserving Frobenius energy while reducing memory bandwidth and arithmetic complexity.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - Automatic rank escalation retries when error threshold is exceeded.
    - Safe fallback to original module if error remains high or decomposition is unstable.
    - Fully vectorized 2:4 structured pruning eliminating Python loop overhead.
    - Robust Conv1D (GPT-2 c_attn, c_proj) and fused linear layer support.
    - Critical-layer protection via should_compress_linear.
    - FP32 computation for all SVD, pruning, and Frobenius norm calculations.
    - Per-layer stats and summary reporting.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 64,
        sparsity: float = 0.5,
        structured_24: bool = True,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        max_retries: int = 2,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="enlighten", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = getattr(cfg, "enlighten_rank", getattr(cfg, "rank", rank))
            sparsity = getattr(cfg, "enlighten_sparsity", getattr(cfg, "sparsity", sparsity))
            structured_24 = getattr(cfg, "structured_24", getattr(cfg, "enlighten_structured_24", structured_24))
            max_error = getattr(cfg, "enlighten_max_error", getattr(cfg, "max_error", max_error))
            fallback_to_original = getattr(
                cfg, "enlighten_fallback_to_original", getattr(cfg, "fallback_to_original", fallback_to_original)
            )

        self.rank = int(rank)
        self.sparsity = float(sparsity)
        self.structured_24 = bool(structured_24)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.max_retries = int(max_retries)
        self.target_modules = target_modules

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_ranks: Dict[str, int] = {}
        self.layer_sparsities: Dict[str, float] = {}
        self.layer_errors: Dict[str, float] = {}

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

    def _decompose_and_prune(
        self, W: torch.Tensor, rank: int
    ) -> Tuple[torch.Tensor, float]:
        """Perform FP32 SVD, compute residual, apply 2:4 / unstructured pruning, and scale."""
        out_f, in_f = W.shape
        r = max(1, min(rank, out_f, in_f))
        norm_w = torch.linalg.norm(W, ord="fro").item()

        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        U_r = U[:, :r]
        S_r = S[:r]
        Vt_r = Vt[:r, :]
        L = (U_r * S_r.unsqueeze(0)) @ Vt_r
        residual = W - L

        if self.structured_24:
            residual_sp = vectorized_24_prune(residual)
        else:
            residual_sp = unstructured_prune(residual, self.sparsity)

        W_recon = L + residual_sp
        recon_norm = torch.linalg.norm(W_recon, ord="fro").item()
        scale = norm_w / (recon_norm + 1e-12) if recon_norm > 1e-12 else 1.0
        W_new = W_recon * scale

        err = (
            float(torch.linalg.norm(W - W_new, ord="fro").item() / max(norm_w, 1e-12))
            if norm_w > 1e-12
            else 0.0
        )
        return W_new, err

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear and Conv1D layers using low-rank SVD + 2:4 structured sparse residuals."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_ranks = {}
        self.layer_sparsities = {}
        self.layer_errors = {}

        targets = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not targets:
            logger.warning("ENLighten: No compressible layers found.")
            return self.model, self.get_compression_stats()

        for name, module in targets:
            cls_name = module.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
            raw_w = module.weight.data

            if is_conv1d:
                W = raw_w.t().float()  # [out_features, in_features]
            else:
                W = raw_w.float()      # [out_features, in_features]

            out_f, in_f = W.shape
            max_r = max(1, min(out_f, in_f))
            current_r = max(1, min(self.rank, max_r))
            best_w_new: Optional[torch.Tensor] = None
            final_err = 1.0

            # Iterative rank retry on error threshold exceedance
            for attempt in range(self.max_retries + 1):
                try:
                    w_cand, err = self._decompose_and_prune(W, current_r)
                    final_err = err
                    best_w_new = w_cand
                    if err <= self.max_error:
                        break
                except Exception as e:
                    logger.debug("ENLighten decomposition failed on %s (iter %d): %s", name, attempt, e)
                    break

                if attempt < self.max_retries:
                    next_r = min(int(current_r * 1.5), max_r)
                    if next_r <= current_r:
                        break
                    logger.debug(
                        "ENLighten: Error %.4f > %.4f on %s. Escalating rank from %d to %d.",
                        err,
                        self.max_error,
                        name,
                        current_r,
                        next_r,
                    )
                    current_r = next_r

            self.layer_ranks[name] = current_r
            self.layer_sparsities[name] = 0.5 if self.structured_24 else self.sparsity
            self.layer_errors[name] = final_err

            if best_w_new is None or final_err > self.max_error:
                logger.warning(
                    "ENLighten reconstruction error %.4f on %s exceeded threshold %.4f.",
                    final_err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping ENLighten compression for %s (fallback to original).", name)
                    self.fallback_layers.append(name)
                    continue

            # Assign compressed weight back in original orientation and dtype
            if is_conv1d:
                module.weight.data.copy_(best_w_new.t().to(dtype=module.weight.dtype, device=module.weight.device))
            else:
                module.weight.data.copy_(best_w_new.to(dtype=module.weight.dtype, device=module.weight.device))

            self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats.update({
            "rank": self.rank,
            "sparsity": 0.5 if self.structured_24 else self.sparsity,
            "structured_24": self.structured_24,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "layer_ranks": self.layer_ranks,
            "layer_sparsities": self.layer_sparsities,
            "layer_errors": self.layer_errors,
            "avg_reconstruction_error": avg_err,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
        })
        logger.info(
            "ENLighten compression complete: %d compressed, %d fallback. Avg error: %.4f.",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("enlighten")
def enlighten_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Enlighten compressor factory function."""
    rank = kwargs.pop("rank", getattr(comp_config, "enlighten_rank", getattr(comp_config, "rank", 64)) if comp_config is not None else 64)
    sparsity = kwargs.pop("sparsity", getattr(comp_config, "enlighten_sparsity", getattr(comp_config, "sparsity", 0.5)) if comp_config is not None else 0.5)
    structured_24 = kwargs.pop("structured_24", getattr(comp_config, "structured_24", True) if comp_config is not None else True)
    max_error = kwargs.pop("max_error", getattr(comp_config, "max_error", 0.20) if comp_config is not None else 0.20)
    fallback = kwargs.pop("fallback_to_original", getattr(comp_config, "fallback_to_original", True) if comp_config is not None else True)

    compressor = ENLightenCompressor(
        model,
        rank=rank,
        sparsity=sparsity,
        structured_24=structured_24,
        max_error=max_error,
        fallback_to_original=fallback,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = ["ENLightenCompressor", "enlighten_factory", "vectorized_24_prune"]
