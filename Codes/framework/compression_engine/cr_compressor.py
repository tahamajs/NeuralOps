"""
CR (Column-Row / Interpolative Decomposition) Compressor.
Decomposes W ≈ C @ P, where C are actual columns of W.
Registered in COMPRESSOR_REGISTRY as 'cr' and 'cr_compressor'.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.core.model_utils import replace_module
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


class CRLinear(nn.Module):
    """
    Interpolative Decomposition / Column-Row Factorized Linear Layer.
    W ≈ C @ P where:
    - C in R^{out_features x rank}: column skeleton from W
    - P in R^{rank x in_features}: interpolative projection matrix

    Forward pass:
    y = ((x @ P^T) @ C^T) + bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))

        self.C = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))
        self.P = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))

        bound = 1.0 / math.sqrt(max(1, in_features))
        nn.init.uniform_(self.C, -bound, bound)
        nn.init.uniform_(self.P, -bound, bound)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return torch.matmul(self.C, self.P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sub-quadratic forward execution:
        y = ((x @ P^T) @ C^T) + bias
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        x_p = F.linear(x_flat, self.P)
        out = F.linear(x_p, self.C, self.bias)

        return out.view(*orig_shape[:-1], self.out_features)

    def get_weight(self) -> torch.Tensor:
        """Materialize full reconstructed weight matrix."""
        return torch.matmul(self.C, self.P)

    @classmethod
    def from_linear(cls, linear: Union[nn.Linear, Any], rank: int = 64) -> "CRLinear":
        """
        Perform Column Interpolative Decomposition on linear or Conv1D layer weight matrix.
        """
        raw_w = linear.weight.data
        bias = linear.bias.data if getattr(linear, "bias", None) is not None else None
        cls_name = linear.__class__.__name__

        is_conv1d = "conv1d" in cls_name.lower() or hasattr(linear, "nf")
        if is_conv1d:
            # Conv1D weights are [in_features, out_features] -> transpose to [out_features, in_features]
            W = raw_w.t().contiguous()
        else:
            W = raw_w.contiguous()

        out_f, in_f = W.shape
        r = max(1, min(rank, out_f, in_f))

        cr_layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=bias is not None,
            dtype=W.dtype,
            device=W.device,
        )

        W_float = W.float()

        # Step 1: Select skeleton columns via QRCP (QR with Column Pivoting) or energy fallback
        col_idx = None
        try:
            import scipy.linalg
            W_np = W_float.cpu().numpy()
            _, _, piv = scipy.linalg.qr(W_np, mode="economic", pivoting=True)
            col_idx = torch.tensor(piv[:r], dtype=torch.long, device=W.device)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
            pass

        if col_idx is None:
            # Energy fallback: select top-r energy columns
            col_norms = (W_float ** 2).sum(dim=0)
            _, col_idx = torch.topk(col_norms, k=r, largest=True, sorted=True)

        C_data = W_float[:, col_idx]  # (out_f, r)

        # Step 2: Tikhonov regularized pseudo-inverse: (C^T C + eps * I)^{-1} C^T
        CtC = torch.matmul(C_data.T, C_data)  # (r, r)
        diag_mean = torch.trace(CtC) / max(1, r)
        eps = max(1e-4 * diag_mean.item(), 1e-4)
        reg_eye = eps * torch.eye(r, device=C_data.device, dtype=C_data.dtype)
        try:
            C_pinv = torch.linalg.solve(CtC + reg_eye, C_data.T)  # (r, out_f)
        except Exception:
            C_pinv = torch.linalg.pinv(C_data, rcond=1e-4)

        P_data = torch.matmul(C_pinv, W_float)  # (r, in_f)

        # Step 3: Symmetrically normalize factor magnitudes to ensure FP16 stability
        norm_C = torch.norm(C_data, p="fro").clamp_min(1e-8)
        norm_P = torch.norm(P_data, p="fro").clamp_min(1e-8)
        scale = torch.sqrt(norm_C / norm_P)
        C_data = C_data / scale
        P_data = P_data * scale

        cr_layer.C.data.copy_(C_data.to(dtype=W.dtype))
        cr_layer.P.data.copy_(P_data.to(dtype=W.dtype))

        if bias is not None and cr_layer.bias is not None:
            cr_layer.bias.data.copy_(bias)

        return cr_layer


@COMPRESSOR_REGISTRY.register("cr_compressor")
class CRCompressor(BaseCompressor):
    """
    CR (Column-Row / Interpolative Decomposition) Compressor.
    Decomposes linear layers into skeleton column bases and interpolative projections.

    Args:
        model: PyTorch model.
        rank: Rank parameter (number of skeleton columns).
        max_error: Maximum relative Frobenius reconstruction error (default 0.20).
        fallback_to_original: If True, keep original layer when error exceeds max_error (default True).
        max_retries: Number of rank increase retries before fallback (default 2).
        cfg: Optional configuration object.
        target_layers: Optional target layer filter.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 64,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        max_retries: int = 2,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="cr", **kwargs)
        self.rank = rank or getattr(cfg, "cr_rank", None) or getattr(cfg, "rank", 64)
        self.max_error = max_error if max_error is not None else getattr(cfg, "max_error", 0.20)
        self.fallback_to_original = fallback_to_original if fallback_to_original is not None else getattr(cfg, "fallback_to_original", True)
        self.max_retries = max_retries
        self.cfg = cfg
        self.target_layers = target_layers or getattr(cfg, "target_layers", None)
        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear layers using interpolative decomposition."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}

        for name, module in list(self.model.named_modules()):
            cls_name = module.__class__.__name__
            is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name.lower() or hasattr(module, "nf")
            if not is_linear_like or isinstance(module, CRLinear):
                continue

            name_lower = name.lower()
            if any(skip in name_lower for skip in ["lm_head", "classifier", "head", "output_layer", "score", "pooler", "embedding"]):
                continue
            if not should_compress_linear(self.model, name):
                continue
            if self.target_layers and not any(t in name for t in self.target_layers):
                continue

            is_conv1d = "conv1d" in cls_name.lower() or hasattr(module, "nf")
            raw_w = module.weight.data
            orig_w = raw_w.t().float() if is_conv1d else raw_w.float()
            out_f, in_f = orig_w.shape
            max_rank = min(out_f, in_f)

            if max_rank < 16:
                continue

            # Parameter reduction check: r * (in_f + out_f) must be less than in_f * out_f
            target_r = min(self.rank, max_rank)
            if target_r * (in_f + out_f) >= 0.85 * (in_f * out_f):
                # Clamp target_r so parameter reduction is guaranteed
                target_r = max(1, int(0.65 * (in_f * out_f) / (in_f + out_f)))

            cur_r = target_r
            best_cr_layer = None
            best_err = float("inf")
            best_r = cur_r
            norm_orig = torch.norm(orig_w, p="fro").clamp_min(1e-8)

            for _ in range(self.max_retries + 1):
                try:
                    candidate = CRLinear.from_linear(module, rank=cur_r)
                    recon_w = candidate.get_weight().float()
                    err = (torch.norm(orig_w - recon_w, p="fro") / norm_orig).item()
                    if err < best_err:
                        best_err = err
                        best_cr_layer = candidate
                        best_r = cur_r

                    if err <= self.max_error:
                        break

                    # Adaptive retry: increase rank if parameter budget allows
                    next_r = min(int(cur_r * 1.5), max_rank - 1)
                    if next_r <= cur_r or next_r * (in_f + out_f) >= 0.85 * (in_f * out_f):
                        break
                    cur_r = next_r
                except Exception as ex:
                    logger.debug("CR decomposition error on %s with rank %d: %s", name, cur_r, ex)
                    break

            if best_err > self.max_error:
                logger.warning(
                    "CRCompressor: Layer %s error %.4f exceeded max_error %.4f",
                    name, best_err, self.max_error
                )
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    continue

            if best_cr_layer is not None:
                replace_module(self.model, name, best_cr_layer)
                self.compressed_layers.append(name)
                self.layer_errors[name] = float(best_err)
                self.layer_ranks[name] = int(best_r)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (sum(self.layer_errors.values()) / len(self.layer_errors)) if self.layer_errors else 0.0
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "compressed_layer_count": len(self.compressed_layers),
                "fallback_layers": self.fallback_layers,
                "fallback_layer_count": len(self.fallback_layers),
                "layer_errors": self.layer_errors,
                "layer_ranks": self.layer_ranks,
                "avg_reconstruction_error": avg_err,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "cr_rank": self.rank,
            }
        )
        return self.model, stats

    def compress_linear(self, linear: Union[nn.Linear, Any]) -> CRLinear:
        """Compress a single linear layer into CRLinear."""
        return CRLinear.from_linear(linear, rank=self.rank)


@COMPRESSOR_REGISTRY.register("cr")
@COMPRESSOR_REGISTRY.register("cr_factory")
@COMPRESSOR_REGISTRY.register("cr_compressor_factory")
def cr_compressor_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for CR compressor."""
    effective_rank = kwargs.pop("rank", rank or getattr(comp_config, "cr_rank", None) or getattr(comp_config, "rank", 64))
    max_error = kwargs.pop("max_error", getattr(comp_config, "max_error", 0.20))
    fallback = kwargs.pop("fallback_to_original", getattr(comp_config, "fallback_to_original", True))
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    compressor = CRCompressor(
        model,
        rank=effective_rank,
        max_error=max_error,
        fallback_to_original=fallback,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model
