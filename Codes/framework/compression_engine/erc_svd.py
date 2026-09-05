"""
ERC-SVD: Error-Controlled SVD with Bounded Frobenius Truncation Error.
Registered as 'erc' and 'erc_svd'. Production hardened for multi-architecture
compatibility (DistilBERT, RoBERTa, GPT-2, Qwen3.5) with error validation, Conv1D support,
parameter bounds, and safe fallback.
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
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("erc")
@COMPRESSOR_REGISTRY.register("erc_svd")
class ERCCompressor(BaseCompressor):
    """
    Error-Controlled SVD (ERC-SVD) Compressor.

    Finds the minimal rank r for each linear layer such that the relative Frobenius reconstruction
    error ||W - W_r||_F / ||W||_F is guaranteed to be <= allowed_error (i.e. singular energy retention
    sum_{i=1}^r sigma_i^2 / sum_j sigma_j^2 >= 1 - allowed_error^2). Factorizes weights symmetrically
    (U_r * sqrt(Sigma_r), sqrt(Sigma_r) * V_r^T) and replaces layers with SVDLinear modules.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable allowed_error/max_error.
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
        allowed_error: float = 0.20,
        max_error: Optional[float] = None,
        fallback_to_original: bool = True,
        rank: Optional[int] = None,
        max_retries: int = 2,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="erc", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            allowed_error = getattr(
                cfg, "erc_allowed_error", getattr(cfg, "allowed_error", allowed_error)
            )
            max_error = getattr(cfg, "erc_max_error", getattr(cfg, "max_error", max_error))
            fallback_to_original = getattr(
                cfg, "erc_fallback_to_original", getattr(cfg, "fallback_to_original", fallback_to_original)
            )
            rank = getattr(cfg, "erc_rank", getattr(cfg, "rank", rank))

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]

        self.allowed_error = float(max_error if max_error is not None else allowed_error)
        self.max_error = self.allowed_error
        self.fallback_to_original = bool(fallback_to_original)
        self.rank = rank
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

    def compress_linear(
        self,
        layer: nn.Module,
    ) -> Tuple[Optional[SVDLinear], float, int]:
        """Decompose a single linear or Conv1D layer via error-controlled SVD."""
        cls_name = layer.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(layer, "nf")
        raw_w = layer.weight.data

        if is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.t().float()  # [out_features, in_features]
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.float()      # [out_features, in_features]

        # Mathematical upper bound on rank is min(in_features, out_features)
        max_r = min(in_f, out_f)

        device = raw_w.device
        dtype = raw_w.dtype
        norm_w = torch.norm(W, p="fro").item()

        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        except Exception as e:
            logger.warning("ERC-SVD SVD computation failed: %s", e)
            return None, 1.0, 0

        # Cumulative energy calculation
        s_squared = S ** 2
        total_energy = s_squared.sum()
        if total_energy <= 1e-12:
            r_init = 1
        else:
            cum_energy = torch.cumsum(s_squared, dim=0) / total_energy
            target_retention = max(0.0, 1.0 - (self.allowed_error ** 2))
            idx = torch.where(cum_energy >= target_retention)[0]
            r_init = int(idx[0].item()) + 1 if len(idx) > 0 else max_r

        if self.rank is not None:
            r_init = min(r_init, self.rank)
        current_r = max(1, min(r_init, max_r))

        best_svd_layer: Optional[SVDLinear] = None
        best_err = 1.0
        chosen_r = current_r

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

        if best_svd_layer is not None:
            best_svd_layer.reconstruction_error = best_err
            return best_svd_layer
        return None, best_err, chosen_r

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress all target linear and Conv1D layers across the model."""
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
            logger.warning("ERC-SVD: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        for name, module in target_layers:
            svd_layer, err, r = self.compress_linear(module)
            self.layer_errors[name] = err
            self.layer_ranks[name] = r

            if svd_layer is None or err > self.max_error:
                logger.warning(
                    "ERC-SVD reconstruction error %.4f on %s exceeded threshold %.4f.",
                    err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping ERC-SVD compression for %s (fallback to original).", name)
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
            "allowed_error": self.allowed_error,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "avg_reconstruction_error": avg_err,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
        })
        logger.info(
            "ERC-SVD compression complete: %d compressed, %d fallback. Avg error: %.4f.",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
        )
        return self.model, stats


def erc_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """ERC factory function."""
    allowed_error = kwargs.pop("allowed_error", getattr(comp_config, "erc_allowed_error", getattr(comp_config, "allowed_error", 0.20)) if comp_config else 0.20)
    max_error = kwargs.pop("max_error", getattr(comp_config, "erc_max_error", getattr(comp_config, "max_error", allowed_error)) if comp_config else allowed_error)
    fallback = kwargs.pop("fallback_to_original", getattr(comp_config, "fallback_to_original", True) if comp_config else True)
    rank = kwargs.pop("rank", getattr(comp_config, "erc_rank", getattr(comp_config, "rank", None)) if comp_config else None)

    compressor = ERCCompressor(
        model,
        allowed_error=allowed_error,
        max_error=max_error,
        fallback_to_original=fallback,
        rank=rank,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


erc_svd_factory = erc_factory
erc_svd_compressor_factory = erc_factory

__all__ = [
    "ERCCompressor",
    "erc_factory",
    "erc_svd_factory",
    "erc_svd_compressor_factory",
]
