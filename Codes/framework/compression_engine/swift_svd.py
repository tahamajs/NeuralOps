"""
Swift-SVD: fast closed-form truncated SVD with symmetric square-root singular value distribution.
Registered as 'swift_svd'.
Production-hardened with:
- Relative Frobenius reconstruction error validation (err <= max_error)
- Adaptive rank retries stepping up rank if error threshold is exceeded
- Safe fallback preserving original layer on failure
- Conv1D (GPT-2) and linear layer support
- Protection of critical classification/embedding/head layers
- Per-layer error logging and stats collection
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import get_dynamic_compression_rank, should_compress_linear

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("swift_svd")
class SwiftSVDCompressor(BaseCompressor):
    """
    Swift-SVD Compressor: Fast Closed-Form Truncated SVD.

    Decomposes dense weight matrices into rank-r factorized layers with symmetric
    square-root singular value splitting (U_r * sqrt(Sigma_r), sqrt(Sigma_r) * V_r^T)
    and replaces nn.Linear/Conv1D with factorized SVDLinear modules.

    Args:
        model: Model to compress.
        rank: Target rank r. If None, computed from rank_ratio.
        rank_ratio: Retention ratio of maximum rank (0.0 < rank_ratio <= 1.0).
        target_modules: Optional list of layer name filters.
        max_error: Maximum allowed relative Frobenius error (default 0.20).
        max_retries: Number of rank-step retries if error exceeds max_error (default 3).
        fallback_to_original: If True, keep original layer if error exceeds max_error.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: Optional[int] = None,
        rank_ratio: float = 0.5,
        target_modules: Optional[List[str]] = None,
        max_error: float = 0.20,
        max_retries: int = 3,
        fallback_to_original: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="swift_svd", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = getattr(cfg, "swift_svd_rank", getattr(cfg, "rank", rank))
            rank_ratio = getattr(cfg, "swift_svd_rank_ratio", getattr(cfg, "rank_ratio", rank_ratio))
            max_error = getattr(cfg, "max_error", getattr(cfg, "max_reconstruction_error", max_error))
            max_retries = getattr(cfg, "max_retries", max_retries)
            fallback_to_original = getattr(cfg, "fallback_to_original", fallback_to_original)

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]
        if "max_error" in kwargs and kwargs["max_error"] is not None:
            max_error = float(kwargs["max_error"])
        if "max_retries" in kwargs and kwargs["max_retries"] is not None:
            max_retries = int(kwargs["max_retries"])
        if "fallback_to_original" in kwargs:
            fallback_to_original = bool(kwargs["fallback_to_original"])

        self.rank = rank
        self.rank_ratio = float(rank_ratio)
        self.target_modules = target_modules
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
        if isinstance(module, SVDLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def compress_linear(
        self,
        layer: nn.Module,
        rank: Optional[int] = None,
        name: str = "",
    ) -> Tuple[Optional[SVDLinear], float, int]:
        """
        Decompose a single linear or Conv1D layer via truncated SVD and return
        (svd_layer_or_None, relative_frobenius_error, final_rank).
        """
        device = layer.weight.device
        dtype = layer.weight.dtype
        raw_w = layer.weight.data.float()

        cls_name = layer.__class__.__name__
        # Handle Conv1D (where weight is (nx, nf) = (in_features, out_features))
        is_conv1d = "conv1d" in cls_name.lower() or (
            raw_w.ndim == 2
            and hasattr(layer, "in_features")
            and layer.in_features == raw_w.shape[0]
            and raw_w.shape[0] < raw_w.shape[1]
        )

        if is_conv1d:
            W = raw_w.t().contiguous()  # (out_features, in_features)
            in_f = W.shape[1]
            out_f = W.shape[0]
        else:
            W = raw_w
            out_f, in_f = W.shape[0], W.shape[1]

        # Bound rank so parameter count r*(in_f + out_f) < in_f*out_f
        max_param_r = max(1, int((in_f * out_f) / (in_f + out_f)) - 1)
        max_r = min(out_f, in_f, max_param_r)
        if max_r < 1:
            return None, 0.0, 0

        # Compute initial target rank
        if rank is not None:
            target_r = max(1, min(int(rank), max_r))
        else:
            target_r = max(1, min(int(round(min(out_f, in_f) * self.rank_ratio)), max_r))

        # Compute full SVD in FP32
        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        except Exception as e:
            try:
                U, S, Vt = torch.linalg.svd(W.cpu(), full_matrices=False)
                U, S, Vt = U.to(device), S.to(device), Vt.to(device)
            except Exception as e2:
                logger.warning("SwiftSVD SVD computation failed for %s: %s. Keeping original module.", name, e2)
                return None, 1.0, 0

        norm_orig = torch.linalg.norm(W).item() + 1e-8
        cur_r = target_r
        best_factors = None
        best_err = float("inf")
        best_r = cur_r

        # Iterative adaptive rank retry if error exceeds max_error
        for attempt in range(self.max_retries + 1):
            sqrt_S = torch.sqrt(torch.clamp(S[:cur_r], min=1e-12))
            U_r = U[:, :cur_r] * sqrt_S.unsqueeze(0)
            V_r = sqrt_S.unsqueeze(1) * Vt[:cur_r, :]

            W_rec = torch.matmul(U_r, V_r)
            rel_err = (torch.linalg.norm(W - W_rec).item()) / norm_orig

            if rel_err < best_err:
                best_err = rel_err
                best_factors = (U_r, V_r)
                best_r = cur_r

            if rel_err <= self.max_error or cur_r >= max_r:
                break

            remaining_retries = max(1, self.max_retries - attempt)
            step = max(4, (max_r - cur_r) // remaining_retries)
            cur_r = min(cur_r + step, max_r)

        # Check threshold
        if best_err > self.max_error:
            logger.warning(
                "SwiftSVD: %s reconstruction error (%.4f) exceeded max_error (%.4f) at rank %d.",
                name,
                best_err,
                self.max_error,
                best_r,
            )
            if self.fallback_to_original:
                return None, best_err, best_r

        if best_factors is None:
            return None, best_err, best_r

        U_final, V_final = best_factors
        U_final = U_final.to(dtype=dtype, device=device)
        V_final = V_final.to(dtype=dtype, device=device)

        has_bias = getattr(layer, "bias", None) is not None
        svd_layer = SVDLinear(
            in_features=in_f,
            out_features=out_f,
            rank=best_r,
            bias=has_bias,
            dtype=dtype,
            device=device,
        )
        svd_layer.set_low_rank_factors(U_final, V_final)

        if has_bias and svd_layer.bias is not None:
            svd_layer.bias.data.copy_(layer.bias.data.to(dtype=dtype, device=device))

        # Dimensional integrity verification
        if svd_layer.in_features != in_f or svd_layer.out_features != out_f:
            logger.error("SwiftSVD: Shape mismatch for %s: expected (%d, %d), got (%d, %d)", name, in_f, out_f, svd_layer.in_features, svd_layer.out_features)
            return None, best_err, best_r

        return svd_layer, best_err, best_r

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress all eligible linear and Conv1D layers in the model."""
        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("SwiftSVD: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        self.compressed_layers = []
        total_error = 0.0

        for name, module in target_layers:
            svd_layer, rel_err, final_rank = self.compress_linear(module, rank=self.rank, name=name)
            if svd_layer is not None:
                replace_module(self.model, name, svd_layer)
                raw_shape = (module.in_features, module.out_features) if hasattr(module, "in_features") and hasattr(module, "out_features") else (svd_layer.in_features, svd_layer.out_features)
                self.compressed_layers.append(
                    {
                        "name": name,
                        "original_shape": raw_shape,
                        "rank": final_rank,
                        "rel_error": float(rel_err),
                    }
                )
                total_error += rel_err
            else:
                logger.info("SwiftSVD: Kept original layer for %s.", name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (total_error / len(self.compressed_layers)) if self.compressed_layers else 0.0
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "num_compressed_layers": len(self.compressed_layers),
                "avg_reconstruction_error": avg_err,
                "rank": self.rank,
                "rank_ratio": self.rank_ratio,
                "max_error": self.max_error,
            }
        )
        logger.info(
            "SwiftSVD compression finished: %d layers compressed, avg error: %.4f, param reduction: %.2f%%",
            len(self.compressed_layers),
            avg_err,
            stats.get("param_reduction_pct", 0.0),
        )
        return self.model, stats


def swift_svd_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """Swift-SVD factory function."""
    compressor = SwiftSVDCompressor(model, comp_config=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model

