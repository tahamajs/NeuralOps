"""
Direct SVD Compressor: Truncated SVD low-rank factorized layer replacement.
Registered as 'direct_svd'. Production hardened for multi-architecture compatibility
(DistilBERT, RoBERTa, GPT-2, Qwen3.5) with error validation, Conv1D support, rank bounds, and safe fallback.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear, get_dynamic_compression_rank

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("direct_svd")
@COMPRESSOR_REGISTRY.register("direct_svd_compressor")
class DirectSVDCompressor(BaseCompressor):
    """
    Direct SVD Compressor.

    Performs direct truncated SVD on target linear and Conv1D layers:
        W ≈ U_r * Sigma_r * V_r^T
    Allocates singular energy symmetrically:
        U_factor = U_r * sqrt(Sigma_r)
        V_factor = sqrt(Sigma_r) * V_r^T
    and replaces dense layers with factorized SVDLinear modules.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - Automatic rank escalation retries when error threshold is exceeded.
    - Safe fallback to original layer if error remains high or decomposition fails.
    - Strict rank safeguard: enforces r < min(in_f, out_f) and parameter reduction r * (in + out) < in * out.
    - Robust Conv1D support (GPT-2 c_attn, c_proj) via weight transposition.
    - Unified critical-layer protection via should_compress_linear.
    - Full FP32 computation for numerical stability.
    - Detailed per-layer statistics and error logging.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: Optional[int] = 32,
        min_rank: int = 16,
        rank_ratio: Optional[float] = None,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        max_retries: int = 2,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="direct_svd", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = getattr(cfg, "direct_svd_rank", getattr(cfg, "svd_rank", getattr(cfg, "rank", rank)))
            min_rank = getattr(cfg, "min_rank", min_rank)
            rank_ratio = getattr(cfg, "direct_svd_rank_ratio", getattr(cfg, "rank_ratio", rank_ratio))
            max_error = getattr(cfg, "direct_svd_max_error", getattr(cfg, "max_error", max_error))
            fallback_to_original = getattr(
                cfg, "direct_svd_fallback_to_original", getattr(cfg, "fallback_to_original", fallback_to_original)
            )

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]

        self.rank = rank
        self.min_rank = max(1, int(min_rank))
        self.rank_ratio = float(rank_ratio) if rank_ratio is not None else None
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.max_retries = int(max_retries)
        self.target_modules = target_modules

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_ranks: Dict[str, int] = {}
        self.layer_errors: Dict[str, float] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        cls_name = module.__class__.__name__.lower()
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
        if not (is_linear or is_conv1d):
            return False
        if isinstance(module, SVDLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def _determine_rank(self, in_f: int, out_f: int, target_r: Optional[int]) -> int:
        """Compute target rank guaranteeing parameter reduction and strict bounds."""
        max_reduction_r = max(1, (in_f * out_f) // (in_f + out_f) - 1)
        max_r = max(1, min(in_f - 1, out_f - 1, max_reduction_r))

        ref_dim = 768.0
        if hasattr(self, "model") and hasattr(self.model, "config") and self.model.config is not None:
            cfg_obj = self.model.config
            ref_dim = float(getattr(cfg_obj, "hidden_size", getattr(cfg_obj, "d_model", getattr(cfg_obj, "n_embd", 768.0))))
        ref_dim = max(1.0, ref_dim)

        if self.rank_ratio is not None:
            r = int(round(min(in_f, out_f) * max(0.01, min(0.99, self.rank_ratio))))
        elif target_r is not None:
            r = int(round(target_r * (min(in_f, out_f) / ref_dim)))
        else:
            r = int(round(32 * (min(in_f, out_f) / ref_dim)))

        # Dynamic architecture-aware rank selection for large models
        r = get_dynamic_compression_rank(in_f, out_f, model_or_family=self.model, default_rank=r, comp_config=getattr(self, "cfg", None))
        r = max(self.min_rank, r)
        return max(1, min(r, max_r))

    def compress_linear(
        self,
        layer: nn.Module,
        rank: Optional[int] = None,
    ) -> Tuple[Optional[SVDLinear], float, int]:
        """
        Decompose a single linear or Conv1D layer via SVD with symmetric factor distribution.
        Returns: (svd_layer, relative_reconstruction_error, chosen_rank)
        """
        cls_name = layer.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(layer, "nf")
        raw_w = layer.weight.data

        if is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.t().float()  # [out_features, in_features]
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.float()      # [out_features, in_features]

        max_reduction_r = max(1, (in_f * out_f) // (in_f + out_f) - 1)
        max_r = max(1, min(in_f - 1, out_f - 1, max_reduction_r))

        current_r = self._determine_rank(in_f, out_f, rank)
        device = raw_w.device
        dtype = raw_w.dtype
        norm_w = torch.norm(W, p="fro").item()

        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        except Exception as e:
            logger.warning("DirectSVD SVD computation failed: %s", e)
            return None, 1.0, 0

        best_svd_layer: Optional[SVDLinear] = None
        best_err = 1.0
        chosen_r = current_r

        # Retry with higher rank if error exceeds threshold
        for attempt in range(self.max_retries + 1):
            r = max(1, min(current_r, max_r))
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
            V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(dtype=dtype, device=device)

            W_recon = torch.matmul(U_r.float(), V_r.float())
            if norm_w < 1e-12:
                err = 0.0
            else:
                err = float(torch.norm(W - W_recon, p="fro").item() / norm_w)

            best_err = err
            chosen_r = r

            svd_layer = SVDLinear(
                in_features=in_f,
                out_features=out_f,
                rank=r,
                bias=layer.bias is not None,
                dtype=dtype,
                device=device,
            )
            svd_layer.set_low_rank_factors(U_r, V_r)
            if layer.bias is not None and svd_layer.bias is not None:
                svd_layer.bias.data.copy_(layer.bias.data)

            best_svd_layer = svd_layer

            if err <= self.max_error or r >= max_r:
                break

            if attempt < self.max_retries:
                next_r = min(int(r * 1.5), max_r)
                if next_r <= r:
                    break
                current_r = next_r

        return best_svd_layer, best_err, chosen_r

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress target linear and Conv1D layers across the model."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_ranks = {}
        self.layer_errors = {}

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("DirectSVD: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        for name, module in target_layers:
            svd_layer, err, r = self.compress_linear(module, rank=self.rank)
            self.layer_errors[name] = err
            self.layer_ranks[name] = r

            if svd_layer is None or err > self.max_error:
                logger.warning(
                    "DirectSVD reconstruction error %.4f on %s exceeded threshold %.4f.",
                    err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping DirectSVD compression for %s (fallback to original).", name)
                    self.fallback_layers.append(name)
                    continue

            replace_module(self.model, name, svd_layer)
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
            "layer_ranks": self.layer_ranks,
            "layer_errors": self.layer_errors,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "avg_reconstruction_error": avg_err,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
        })
        logger.info(
            "DirectSVD compression complete: %d compressed, %d fallback. Avg error: %.4f.",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
        )
        return self.model, stats


def direct_svd_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """Direct SVD factory function."""
    rank = getattr(comp_config, "direct_svd_rank", getattr(comp_config, "rank", kwargs.get("rank", 32)))
    max_error = getattr(comp_config, "direct_svd_max_error", getattr(comp_config, "max_error", kwargs.get("max_error", 0.20)))
    fallback = getattr(
        comp_config, "direct_svd_fallback_to_original", getattr(comp_config, "fallback_to_original", kwargs.get("fallback_to_original", True))
    )
    compressor = DirectSVDCompressor(
        model,
        rank=rank,
        max_error=max_error,
        fallback_to_original=fallback,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


direct_svd_compressor_factory = direct_svd_factory

__all__ = ["DirectSVDCompressor", "direct_svd_factory", "direct_svd_compressor_factory"]
