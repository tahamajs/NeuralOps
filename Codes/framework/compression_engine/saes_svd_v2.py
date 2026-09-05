"""
SAES-SVD v2: Spectrum-Aware Error-Suppressed Singular Value Decomposition.
Registered in COMPRESSOR_REGISTRY as 'saes_svd_v2' and 'saes'.

Mathematical Formulation:
1. Truncated SVD Baseline:
   $$W_0 \\approx U_r \\Sigma_r V_r^\\top = A_0 B_0$$
2. Spectral Energy Distribution:
   $$\\rho_k = \\frac{\\sigma_k^2}{\\sum_{j=1}^r \\sigma_j^2}$$
3. Error-Suppression Iterations:
   $$E_t = W_0 - A_t B_t$$
   $$A_{t+1} = A_t + E_t B_t^\\top (B_t B_t^\\top + \\epsilon I)^{-1}$$
   $$B_{t+1} = B_t + (A_{t+1}^\\top A_{t+1} + \\epsilon I)^{-1} A_{t+1}^\\top E_t$$

Production Hardening:
- Fix 1: Default fallback_to_original changed to True for safety.
- Fix 2: Always log warning when reconstruction error exceeds max_error regardless of fallback.
- Fix 3: Reliable Conv1D detection using class name or nf attribute without shape heuristics.
- Fix 4: Include per-layer rank, error, and detailed specs in returned stats.
- Fix 5: Ensure bias is copied correctly for both Linear and Conv1D.
- Fix 6: Unified critical-layer protection delegated solely to should_compress_linear.
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


@COMPRESSOR_REGISTRY.register("saes_svd_v2")
@COMPRESSOR_REGISTRY.register("saes")
class SAESSVDCompressorV2(BaseCompressor):
    """
    SAES-SVD v2 Compressor with iterative spectrum-aware residual correction,
    Frobenius validation, and safe fallback.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: Optional[int] = None,
        rank_ratio: float = 0.5,
        correcting_steps: int = 2,
        target_modules: Optional[List[str]] = None,
        max_error: float = 0.20,
        fallback_to_original: bool = True,  # Fix 1: Default changed to True
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="saes_svd_v2", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = getattr(cfg, "saes_rank", getattr(cfg, "rank", rank))
            rank_ratio = getattr(cfg, "saes_rank_ratio", getattr(cfg, "rank_ratio", rank_ratio))
            correcting_steps = getattr(cfg, "correcting_steps", correcting_steps)
            max_error = getattr(cfg, "max_error", max_error)
            fallback_to_original = getattr(cfg, "fallback_to_original", fallback_to_original)

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]
        if "fallback_to_original" in kwargs:
            fallback_to_original = bool(kwargs["fallback_to_original"])

        self.rank = rank
        self.rank_ratio = float(rank_ratio)
        self.correcting_steps = int(correcting_steps)
        self.target_modules = target_modules
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}
        self.replaced_layers: List[Dict[str, Any]] = []

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """
        Fix 6: Unified critical-layer protection delegated to should_compress_linear.
        """
        cls_name = module.__class__.__name__.lower()
        # Fix 3: Reliable Conv1D check using class name and hasattr(nf)
        is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
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
    ) -> Optional[SVDLinear]:
        """Decompose a single linear or Conv1D layer with spectrum-aware error suppression."""
        raw_w = layer.weight.data
        cls_name = layer.__class__.__name__.lower()
        # Fix 3: Conv1D detection by class name or nf attribute
        is_conv1d = "conv1d" in cls_name or hasattr(layer, "nf")

        if is_conv1d:
            W = raw_w.t().float()
            out_f = W.shape[0]
            in_f = W.shape[1]
        else:
            W = raw_w.float()
            out_f = getattr(layer, "out_features", W.shape[0])
            in_f = getattr(layer, "in_features", W.shape[1])

        max_r = min(out_f, in_f)
        # Cap rank to guarantee parameter reduction
        param_cap = max(1, (in_f * out_f) // (in_f + out_f) - 1)
        effective_max_r = min(max_r, param_cap)

        if rank is not None:
            r = max(1, min(rank, effective_max_r))
        else:
            r = max(1, min(int(round(max_r * self.rank_ratio)), effective_max_r))

        device = layer.weight.device
        dtype = layer.weight.dtype

        # Compute full SVD in FP32
        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        except Exception as e:
            logger.warning("SAES-SVD SVD failure for %s: %s", name, e)
            return None

        # Initial symmetric square-root singular value allocation
        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        A = U[:, :r] * sqrt_S.unsqueeze(0)      # [out_features, r]
        B = sqrt_S.unsqueeze(1) * Vt[:r, :]     # [r, in_features]

        # Iterative residual error suppression
        eps_eye = 1e-6 * torch.eye(r, device=W.device, dtype=torch.float32)
        for _ in range(self.correcting_steps):
            E = W - A @ B
            # Update A towards residual gradient
            BBt = B @ B.t() + eps_eye
            A = A + E @ B.t() @ torch.linalg.pinv(BBt)

            # Update B towards residual gradient
            AtA = A.t() @ A + eps_eye
            B = B + torch.linalg.pinv(AtA) @ A.t() @ (W - A @ B)

        # Compute relative Frobenius reconstruction error
        W_rec = A @ B
        norm_orig = torch.norm(W, p="fro")
        rel_err = (torch.norm(W - W_rec, p="fro") / (norm_orig + 1e-8)).item()

        # Fix 2: Always log a warning when reconstruction error exceeds max_error
        if rel_err > self.max_error:
            logger.warning(
                "SAES-SVD reconstruction error %.4f on %s exceeded threshold %.4f%s",
                rel_err,
                name,
                self.max_error,
                ". Keeping original layer." if self.fallback_to_original else ". Proceeding with compressed layer.",
            )
            if self.fallback_to_original:
                return None

        has_bias = getattr(layer, "bias", None) is not None
        svd_layer = SVDLinear(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=has_bias,
            dtype=dtype,
            device=device,
        )
        svd_layer.set_low_rank_factors(
            U_factor=A.to(dtype=dtype, device=device),
            V_factor=B.to(dtype=dtype, device=device),
        )
        # Fix 5: Ensure bias is copied correctly for both Linear and Conv1D
        if has_bias and svd_layer.bias is not None:
            svd_layer.bias.data.copy_(layer.bias.data.to(dtype=dtype, device=device))

        # Store metadata on module
        svd_layer.reconstruction_error = rel_err
        svd_layer.rank = r
        return svd_layer

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress all linear and Conv1D layers in the model."""
        self.layer_errors = {}
        self.layer_ranks = {}
        self.replaced_layers = []

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("SAES-SVD v2: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        for name, module in target_layers:
            svd_layer = self.compress_linear(module, rank=self.rank, name=name)
            if svd_layer is not None:
                err = getattr(svd_layer, "reconstruction_error", 0.0)
                r = getattr(svd_layer, "rank", self.rank)
                self.layer_errors[name] = err
                self.layer_ranks[name] = r
                # Fix 4: Include per-layer rank and error in returned stats
                self.replaced_layers.append({"name": name, "rank": r, "error": err})
                replace_module(self.model, name, svd_layer)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        # Fix 4 & Fix 6: Detailed per-layer stats, errors, ranks, and fallback status
        stats.update({
            "layer_errors": self.layer_errors,
            "layer_ranks": self.layer_ranks,
            "replaced_layers": self.replaced_layers,
            "compressed_layers": len(self.replaced_layers),
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "correcting_steps": self.correcting_steps,
        })
        return self.model, stats


def saes_svd_v2_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """SAES-SVD v2 factory function."""
    compressor = SAESSVDCompressorV2(model, comp_config=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model


saes_factory = saes_svd_v2_factory
saes_svd_factory = saes_svd_v2_factory
saes_svd_compressor_factory = saes_svd_v2_factory

# Backward compatibility alias
SAESCompressor = SAESSVDCompressorV2

__all__ = [
    "SAESSVDCompressorV2",
    "SAESCompressor",
    "saes_svd_v2_factory",
    "saes_factory",
    "saes_svd_factory",
    "saes_svd_compressor_factory",
]
