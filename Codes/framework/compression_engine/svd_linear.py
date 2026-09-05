"""
SVD-based low-rank linear layer replacement with optional Fused SVD kernel acceleration.
Registered as 'svd_linear' and 'svd_linear_compressor'.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.attentions.fused_svd_kernel import FusedSVDLinear, fused_svd_linear
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


class SVDLinearLayer(nn.Module):
    """
    Replace a linear layer with factorized U @ V, keeping top-`rank` components.
    Optionally dispatches fused in-SRAM kernel to eliminate memory roundtrips.
    """

    def __init__(
        self,
        linear_layer: nn.Module,
        rank: int,
        use_fused_kernel: bool = False,
        use_rms_norm: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        raw_w = linear_layer.weight.data
        cls_name = linear_layer.__class__.__name__
        is_conv1d = "conv1d" in cls_name.lower() or (
            raw_w.ndim == 2
            and hasattr(linear_layer, "in_features")
            and linear_layer.in_features == raw_w.shape[0]
            and raw_w.shape[0] < raw_w.shape[1]
        )

        if is_conv1d:
            W = raw_w.t().float()
            self.out_features = W.shape[0]
            self.in_features = W.shape[1]
        else:
            W = raw_w.float()
            self.out_features = getattr(linear_layer, "out_features", W.shape[0])
            self.in_features = getattr(linear_layer, "in_features", W.shape[1])

        param_cap = max(1, (self.in_features * self.out_features) // (self.in_features + self.out_features) - 1)
        self.rank: int = max(1, min(rank, self.in_features, self.out_features, param_cap))
        self.use_fused_kernel: bool = use_fused_kernel
        self.use_rms_norm: bool = use_rms_norm
        self.eps: float = eps

        device = linear_layer.weight.device
        dtype = linear_layer.weight.dtype

        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            sqrt_s = torch.sqrt(torch.clamp(S[: self.rank], min=1e-12))
            U_weighted = (U[:, : self.rank] * sqrt_s.unsqueeze(0)).to(device=device, dtype=dtype)
            V_weighted = (sqrt_s.unsqueeze(1) * Vh[: self.rank, :]).to(device=device, dtype=dtype)

        self.U = nn.Parameter(U_weighted)
        self.V = nn.Parameter(V_weighted)

        if use_rms_norm:
            self.norm_weight = nn.Parameter(torch.ones(self.rank, dtype=dtype, device=device))
        else:
            self.register_parameter("norm_weight", None)

        if linear_layer.bias is not None:
            self.bias = nn.Parameter(linear_layer.bias.data.clone())
        else:
            self.register_parameter("bias", None)

        # Frobenius reconstruction error
        W_rec = torch.matmul(U_weighted.float(), V_weighted.float())
        norm_orig = torch.norm(W, p="fro")
        self.reconstruction_error = (torch.norm(W - W_rec, p="fro") / (norm_orig + 1e-8)).item()

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        return torch.matmul(self.U, self.V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional fused kernel acceleration."""
        if self.use_fused_kernel:
            return fused_svd_linear(
                x=x,
                V=self.V,
                U=self.U,
                norm_weight=self.norm_weight if self.use_rms_norm else None,
                bias=self.bias,
                eps=self.eps,
                use_rms_norm=self.use_rms_norm,
            )

        latent = F.linear(x, self.V)
        out = F.linear(latent, self.U, self.bias)
        return out


@COMPRESSOR_REGISTRY.register("svd_linear")
@COMPRESSOR_REGISTRY.register("svd_linear_compressor")
class SVDLinearCompressor(BaseCompressor):
    """Compresses model by replacing Linear/Conv1D layers with SVDLinearLayer."""

    def __init__(
        self,
        model: nn.Module,
        rank: int = 64,
        max_error: float = 0.5,
        fallback_to_original: bool = False,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="svd_linear", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = getattr(cfg, "svd_rank", getattr(cfg, "rank", rank))
            max_error = getattr(cfg, "max_error", max_error)
            fallback_to_original = getattr(cfg, "fallback_to_original", fallback_to_original)

        self.rank = int(rank)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_modules = target_modules
        self.layer_errors: Dict[str, float] = {}
        self.replaced_layers: List[Dict[str, Any]] = []

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if any(k in name.lower() for k in ["classifier", "score", "lm_head", "pooler", "embedding"]):
            return False
        cls_name = module.__class__.__name__
        is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name.lower()
        if not is_linear_like:
            return False
        if isinstance(module, SVDLinearLayer):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear layers with SVD factorization."""
        self.layer_errors = {}
        self.replaced_layers = []

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("SVDLinear: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        for name, module in target_layers:
            try:
                new_layer = SVDLinearLayer(module, rank=self.rank)
                err = new_layer.reconstruction_error
                self.layer_errors[name] = err

                if self.fallback_to_original and err > self.max_error:
                    logger.warning(
                        "SVDLinear reconstruction error %.4f on %s exceeded threshold %.4f. Skipping.",
                        err,
                        name,
                        self.max_error,
                    )
                    continue

                self.replaced_layers.append({"name": name, "rank": new_layer.rank, "error": err})
                replace_module(self.model, name, new_layer)
            except Exception as e:
                logger.warning("Failed to compress %s with SVDLinear: %s", name, e)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats["layer_errors"] = self.layer_errors
        stats["compressed_layers"] = len(self.replaced_layers)
        return self.model, stats


def svd_linear_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """SVD Linear compressor factory."""
    rank = kwargs.pop("rank", getattr(comp_config, "rank", 64) if comp_config is not None else 64)
    compressor = SVDLinearCompressor(model, rank=rank, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = [
    "SVDLinearLayer",
    "SVDLinearCompressor",
    "svd_linear_factory",
    "FusedSVDLinear",
    "fused_svd_linear",
]
