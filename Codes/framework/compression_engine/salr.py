"""
SALR: Structure-Aware / Sparse Low-Rank compression combining SVD with bottleneck pruning.
Registered as 'salr'.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


class SALRLayer(nn.Module):
    """
    Structure-Aware Low-Rank + Sparse Layer (SALR).

    Combines truncated SVD rank reduction with sparse singular component selection:
        y = (x @ (V * mask).T) @ U.T + bias
    where U: [out_features, rank], V: [rank, in_features], mask: [rank, 1].
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        sparsity: float = 0.5,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))
        self.sparsity = float(sparsity)

        self.V = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))
        self.U = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        # 1D binary mask on singular bottleneck components: [rank, 1]
        mask = torch.ones(self.rank, 1, dtype=torch.float32, device=device)
        if 0.0 < self.sparsity < 1.0:
            num_zeros = int(round(self.rank * self.sparsity))
            # Zero out the smallest/trailing components
            if num_zeros > 0:
                mask[-num_zeros:, :] = 0.0
        self.register_buffer("mask", mask)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) -> None:
        with torch.no_grad():
            self.U.copy_(U_factor)
            self.V.copy_(V_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Two-stage forward pass with sparse bottleneck mask."""
        V_masked = self.V * self.mask.to(dtype=self.V.dtype)
        latent = F.linear(x, V_masked)              # [..., rank]
        out = F.linear(latent, self.U, self.bias)  # [..., out_features]
        return out

    @property
    def weight(self) -> torch.Tensor:
        """Effective reconstructed weight matrix."""
        V_masked = self.V * self.mask.to(dtype=self.V.dtype)
        return torch.matmul(self.U, V_masked)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, sparsity={self.sparsity}"


@COMPRESSOR_REGISTRY.register("salr")
class SALRCompressor(BaseCompressor):
    """
    SALR Compressor: Structure-Aware Low-Rank and Sparse Pruning.

    Factorizes dense nn.Linear or Conv1D layers into symmetric low-rank factors
    (U * sqrt(Sigma), sqrt(Sigma) * V^T) and applies structured sparse pruning
    to trailing singular components in the bottleneck.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 64,
        sparsity: float = 0.5,
        target_modules: Optional[list[str]] = None,
        max_error: float = 0.5,
        fallback_to_original: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="salr", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = getattr(cfg, "salr_rank", getattr(cfg, "rank", rank))
            sparsity = getattr(cfg, "salr_sparsity", getattr(cfg, "sparsity", sparsity))
            max_error = getattr(cfg, "max_error", max_error)
            fallback_to_original = getattr(cfg, "fallback_to_original", fallback_to_original)

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]

        self.rank = int(rank)
        self.sparsity = float(sparsity)
        self.target_modules = target_modules
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.layer_errors: Dict[str, float] = {}
        self.replaced_layers: List[Dict[str, Any]] = []

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if any(k in name.lower() for k in ["classifier", "score", "lm_head", "pooler", "embedding"]):
            return False
        cls_name = module.__class__.__name__
        is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name.lower()
        if not is_linear_like:
            return False
        if isinstance(module, SALRLayer):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def compress_linear(
        self,
        layer: nn.Module,
        rank: int,
    ) -> Optional[SALRLayer]:
        """Decompose a single linear or Conv1D layer into SALRLayer with validation."""
        raw_w = layer.weight.data
        cls_name = layer.__class__.__name__
        is_conv1d = "conv1d" in cls_name.lower() or (
            raw_w.ndim == 2
            and hasattr(layer, "in_features")
            and layer.in_features == raw_w.shape[0]
            and raw_w.shape[0] < raw_w.shape[1]
        )

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
        r = max(1, min(rank, max_r, param_cap))

        device = layer.weight.device
        dtype = layer.weight.dtype

        # SVD in FP32
        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        except Exception as e:
            logger.warning("SALR SVD failure: %s", e)
            return None

        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
        V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(dtype=dtype, device=device)

        salr_layer = SALRLayer(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            sparsity=self.sparsity,
            bias=layer.bias is not None,
            dtype=dtype,
            device=device,
        )
        salr_layer.set_low_rank_factors(U_r, V_r)
        if layer.bias is not None:
            salr_layer.bias.data.copy_(layer.bias.data)

        # Frobenius reconstruction error check
        W_rec = salr_layer.weight.float()
        norm_orig = torch.norm(W, p="fro")
        rel_err = (torch.norm(W - W_rec, p="fro") / (norm_orig + 1e-8)).item()

        if self.fallback_to_original and rel_err > self.max_error:
            logger.warning(
                "SALR reconstruction error %.4f exceeded threshold %.4f. Keeping original layer.",
                rel_err,
                self.max_error,
            )
            return None

        salr_layer.reconstruction_error = rel_err
        return salr_layer

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress all linear layers in the model."""
        self.layer_errors = {}
        self.replaced_layers = []

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("SALR: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        for name, module in target_layers:
            new_layer = self.compress_linear(module, rank=self.rank)
            if new_layer is not None:
                err = getattr(new_layer, "reconstruction_error", 0.0)
                self.layer_errors[name] = err
                self.replaced_layers.append({"name": name, "rank": new_layer.rank, "error": err})
                replace_module(self.model, name, new_layer)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats["layer_errors"] = self.layer_errors
        stats["compressed_layers"] = len(self.replaced_layers)
        return self.model, stats


def salr_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """SALR factory function."""
    compressor = SALRCompressor(model, comp_config=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = [
    "SALRLayer",
    "SALRCompressor",
    "salr_factory",
]
