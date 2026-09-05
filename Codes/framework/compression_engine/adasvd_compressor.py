"""
AdaSVD: Adaptive SVD with per-layer importance allocation and iterative ALS refinement.
Registered as 'adasvd'. Production hardened with reconstruction validation, fallback,
Conv1D support, rank reduction bounds, and comprehensive per-layer stats.
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


@COMPRESSOR_REGISTRY.register("adasvd")
class AdaSVDCompressor(BaseCompressor):
    """
    Adaptive SVD Compressor (AdaSVD).

    Assigns SVD ranks dynamically across layers proportional to spectral importance
    (nuclear norm ||W||_* = sum(sigma_i)), initializes factorized matrices with
    symmetric square-root singular value distribution (A = U * sqrt(Sigma), B = sqrt(Sigma) * V^T),
    refines factors via Alternating Least Squares (ALS), and replaces dense layers
    with factorized SVDLinear modules.

    Hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - ALS check vs initial SVD: reverts to initial SVD factors if ALS worsens error.
    - Safe fallback to original layer if error exceeds max_error or factorization fails.
    - Enforced rank safeguard: r < min(in_features, out_features) and r * (in + out) < in * out.
    - Robust Conv1D support (GPT-2 c_attn, c_proj) via weight transposition and SVDLinear replacement.
    - Single unified critical-layer protection via should_compress_linear.
    - Full FP32 computation for numerical stability, casting back to original dtype at module insertion.
    - Per-layer logging and detailed compression statistics.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: Optional[int] = None,
        target_param_ratio: float = 0.7,
        num_iters: int = 5,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="adasvd", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = getattr(cfg, "adasvd_rank", getattr(cfg, "rank", rank))
            target_param_ratio = getattr(
                cfg, "adasvd_target_ratio", getattr(cfg, "target_param_ratio", target_param_ratio)
            )
            num_iters = getattr(cfg, "adasvd_iters", getattr(cfg, "num_iters", num_iters))
            max_error = getattr(cfg, "adasvd_max_error", getattr(cfg, "max_error", max_error))
            fallback_to_original = getattr(
                cfg, "adasvd_fallback_to_original", getattr(cfg, "fallback_to_original", fallback_to_original)
            )

        self.rank = rank
        self.target_param_ratio = float(target_param_ratio)
        self.num_iters = int(num_iters)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_modules = target_modules

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}
        self.layer_iters: Dict[str, int] = {}

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

    @staticmethod
    def _compute_nuclear_norm(W: torch.Tensor) -> float:
        """Compute nuclear norm ||W||_* = sum(sigma_i) in FP32."""
        with torch.no_grad():
            s = torch.linalg.svdvals(W.float())
            return float(s.sum().item())

    def compress_linear(
        self,
        layer: nn.Module,
        rank: int,
    ) -> Tuple[Optional[SVDLinear], float, int]:
        """
        Compress a single nn.Linear or Conv1D layer using symmetric SVD and ALS refinement.
        Returns: (svd_layer, relative_reconstruction_error, actual_als_iters)
        """
        cls_name = layer.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(layer, "nf")
        raw_w = layer.weight.data

        if is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.t().float()  # [out_features, in_features]
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.float()  # [out_features, in_features]

        # Enforce rank < min(in_f, out_f) and parameter reduction: r * (in_f + out_f) < in_f * out_f
        max_param_r = max(1, (in_f * out_f) // (in_f + out_f) - 1)
        max_r = max(1, min(in_f - 1, out_f - 1, max_param_r))
        r = max(1, min(rank, max_r))
        device = layer.weight.device
        dtype = layer.weight.dtype

        # Compute truncated SVD in FP32
        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        except Exception as e:
            logger.warning("AdaSVD SVD computation failed: %s", e)
            return None, 1.0, 0

        # Symmetric square-root singular value initialization
        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        A_0 = U[:, :r] * sqrt_S.unsqueeze(0)       # [out_features, r]
        B_0 = sqrt_S.unsqueeze(1) * Vt[:r, :]      # [r, in_features]

        W_norm = torch.norm(W, p="fro").item()
        if W_norm < 1e-12:
            init_err = 0.0
        else:
            recon_0 = torch.matmul(A_0, B_0)
            init_err = float(torch.norm(W - recon_0, p="fro").item() / W_norm)

        A = A_0.clone()
        B = B_0.clone()
        actual_iters = 0

        # Alternating Least Squares (ALS) refinement
        eps_eye = 1e-6 * torch.eye(r, device=device, dtype=torch.float32)
        for it in range(self.num_iters):
            BBt = B @ B.t() + eps_eye
            try:
                A_next = W @ B.t() @ torch.linalg.inv(BBt)
                AtA = A_next.t() @ A_next + eps_eye
                B_next = torch.linalg.inv(AtA) @ A_next.t() @ W
            except Exception as e:
                logger.debug("AdaSVD ALS inversion failed at iter %d: %s", it, e)
                break

            if (
                torch.isnan(A_next).any()
                or torch.isnan(B_next).any()
                or torch.isinf(A_next).any()
                or torch.isinf(B_next).any()
            ):
                logger.debug("AdaSVD NaN/Inf during ALS iter %d. Halting ALS.", it)
                break

            A, B = A_next, B_next
            actual_iters += 1

        # Check ALS factors validity and error vs initial SVD
        if (
            torch.isnan(A).any()
            or torch.isnan(B).any()
            or torch.isinf(A).any()
            or torch.isinf(B).any()
        ):
            A, B = A_0, B_0
            final_err = init_err
        else:
            recon = torch.matmul(A, B)
            if W_norm < 1e-12:
                final_err = 0.0
            else:
                final_err = float(torch.norm(W - recon, p="fro").item() / W_norm)

            if final_err > init_err:
                logger.debug(
                    "AdaSVD: ALS error (%.4f) > initial SVD error (%.4f). Reverting to initial SVD.",
                    final_err,
                    init_err,
                )
                A, B = A_0, B_0
                final_err = init_err

        # Instantiate structural SVDLinear
        svd_layer = SVDLinear(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=layer.bias is not None,
            dtype=dtype,
            device=device,
        )
        svd_layer.set_low_rank_factors(
            U_factor=A.to(dtype=dtype, device=device),
            V_factor=B.to(dtype=dtype, device=device),
        )
        if layer.bias is not None and svd_layer.bias is not None:
            svd_layer.bias.data.copy_(layer.bias.data)

        return svd_layer, final_err, actual_iters

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Executes adaptive rank assignment and ALS refinement across all target linear/conv1d layers.
        """
        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        self.layer_iters = {}

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("AdaSVD: No compressible layers found.")
            return self.model, self.get_compression_stats()

        # Step 1: Compute spectral nuclear norms in FP32
        layer_importances = {}
        total_importance = 0.0
        for name, module in target_layers:
            cls_name = module.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
            raw_w = module.weight.data
            W_2d = raw_w.t().float() if is_conv1d else raw_w.float()
            imp = self._compute_nuclear_norm(W_2d)
            layer_importances[name] = max(imp, 1e-8)
            total_importance += layer_importances[name]

        # Step 2: Assign ranks per layer and compress
        for name, module in target_layers:
            cls_name = module.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
            raw_w = module.weight.data
            if is_conv1d:
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            else:
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]

            max_param_r = max(1, (in_f * out_f) // (in_f + out_f) - 1)
            max_r = max(1, min(in_f - 1, out_f - 1, max_param_r))

            if self.rank is not None:
                frac = layer_importances[name] / (total_importance / len(target_layers))
                allocated_rank = int(round(self.rank * frac))
            else:
                frac = layer_importances[name] / total_importance
                allocated_rank = int(round(min(in_f, out_f) * self.target_param_ratio * frac * len(target_layers)))

            allocated_rank = max(1, min(allocated_rank, max_r))

            svd_layer, err, iters = self.compress_linear(module, rank=allocated_rank)
            self.layer_errors[name] = err
            self.layer_ranks[name] = allocated_rank
            self.layer_iters[name] = iters

            if svd_layer is None or err > self.max_error:
                logger.warning(
                    "AdaSVD reconstruction error %.4f on %s exceeded threshold %.4f.",
                    err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping AdaSVD compression for %s (fallback to original).", name)
                    self.fallback_layers.append(name)
                    continue

            replace_module(self.model, name, svd_layer)
            self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "layer_errors": self.layer_errors,
            "layer_ranks": self.layer_ranks,
            "layer_iters": self.layer_iters,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
        })
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats["avg_reconstruction_error"] = avg_err
        logger.info(
            "AdaSVD compression complete: %d compressed, %d fallback. Avg error: %.4f.",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
        )
        return self.model, stats


def adasvd_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """AdaSVD factory function."""
    compressor = AdaSVDCompressor(model, comp_config=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model


adasvd_compressor_factory = adasvd_factory

__all__ = ["AdaSVDCompressor", "adasvd_factory", "adasvd_compressor_factory"]
