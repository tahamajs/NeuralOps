"""
NMF: Non-negative Matrix Factorization Linear Compressor.
Registered in COMPRESSOR_REGISTRY as 'nmf'.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class NMFLinear(nn.Module):
    """
    Linear layer decomposed via Non-negative Matrix Factorization (NMF):
    W = H @ V + shift
    - H in R_+^{out_features x rank}
    - V in R_+^{rank x in_features}
    - shift in R

    Forward pass:
    y = ((x @ V^T) @ H^T) + shift * sum(x, dim=-1, keepdim=True) + bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        shift: float = 0.0,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)
        self.register_buffer("shift", torch.tensor(shift, dtype=dtype or torch.float32, device=device))

        self.H = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))
        self.V = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))

        bound = 1.0 / math.sqrt(max(1, in_features))
        nn.init.uniform_(self.H, 0.0, bound)
        nn.init.uniform_(self.V, 0.0, bound)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return torch.matmul(self.H, self.V) + self.shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sub-quadratic forward pass using factorized matrix multiplications.
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # Factorized forward: (x @ V^T) @ H^T
        x_v = F.linear(x_flat, self.V)  # (K, rank)
        out = F.linear(x_v, self.H)     # (K, out_features)

        # Account for constant baseline shift if non-zero
        if self.shift.abs() > 1e-8:
            x_sum = x_flat.sum(dim=-1, keepdim=True)  # (K, 1)
            out = out + x_sum * self.shift

        if self.bias is not None:
            out = out + self.bias

        return out.view(*orig_shape[:-1], self.out_features)

    def get_weight(self) -> torch.Tensor:
        """Materialize full reconstructed weight matrix."""
        return torch.matmul(self.H, self.V) + self.shift

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        rank: int = 64,
        max_iter: int = 100,
    ) -> "NMFLinear":
        """
        Fit Non-negative Matrix Factorization on linear or Conv1D layer weights using
        PyTorch Multiplicative Updates (Lee & Seung 2001) or sklearn NMF.
        """
        raw_w = linear.weight.data
        cls_name = linear.__class__.__name__
        is_conv1d = "conv1d" in cls_name.lower() or (
            raw_w.ndim == 2
            and hasattr(linear, "in_features")
            and linear.in_features == raw_w.shape[0]
            and raw_w.shape[0] < raw_w.shape[1]
        )

        if is_conv1d:
            W = raw_w.t().contiguous()
            out_f, in_f = W.shape
        else:
            W = raw_w
            out_f, in_f = W.shape

        bias = linear.bias.data if getattr(linear, "bias", None) is not None else None
        max_r = min(out_f, in_f)
        # Cap rank to guarantee parameter reduction
        param_cap = max(1, (in_f * out_f) // (in_f + out_f) - 1) if (in_f * out_f > 16) else max_r
        r = max(1, min(rank, max_r, param_cap))

        min_val = W.min().item()
        shift_val = min_val - 1e-5 if min_val < 0 else 0.0
        W_pos = (W.float() - shift_val).clamp_min(1e-6)

        fitted = False
        H_init: Optional[torch.Tensor] = None
        V_init: Optional[torch.Tensor] = None

        try:
            import warnings
            from sklearn.decomposition import NMF as skNMF
            from sklearn.exceptions import ConvergenceWarning

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                sk_model = skNMF(
                    n_components=r,
                    init="nndsvda",
                    max_iter=max_iter,
                    tol=1e-4,
                    random_state=42,
                )
                H_np = sk_model.fit_transform(W_pos.cpu().numpy())
                V_np = sk_model.components_
                H_init = torch.tensor(H_np, dtype=W.dtype, device=W.device)
                V_init = torch.tensor(V_np, dtype=W.dtype, device=W.device)
                fitted = True
        except Exception:
            fitted = False

        if not fitted or H_init is None or V_init is None:
            U, S, Vt = torch.linalg.svd(W_pos, full_matrices=False)
            H_init = (U[:, :r].abs() * torch.sqrt(S[:r].unsqueeze(0))).clamp_min(1e-4)
            V_init = (torch.sqrt(S[:r].unsqueeze(1)) * Vt[:r, :].abs()).clamp_min(1e-4)

            for _ in range(max_iter):
                HT = H_init.t()
                HTW = torch.matmul(HT, W_pos)
                HTHV = torch.matmul(torch.matmul(HT, H_init), V_init).clamp_min(1e-10)
                V_init = V_init * (HTW / HTHV)

                VT = V_init.t()
                WVT = torch.matmul(W_pos, VT)
                HVVT = torch.matmul(torch.matmul(H_init, V_init), VT).clamp_min(1e-10)
                H_init = H_init * (WVT / HVVT)

        nmf_layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            shift=shift_val,
            bias=bias is not None,
            dtype=W.dtype,
            device=W.device,
        )

        nmf_layer.H.data.copy_(H_init.to(dtype=W.dtype))
        nmf_layer.V.data.copy_(V_init.to(dtype=W.dtype))

        if bias is not None and nmf_layer.bias is not None:
            nmf_layer.bias.data.copy_(bias)

        return nmf_layer


class NMFCompressor(BaseCompressor):
    """
    Non-negative Matrix Factorization (NMF) Compressor.
    Decomposes linear layer weights into low-rank non-negative factors.

    Args:
        model: Model to compress.
        rank: Target rank for NMF factorization.
        max_iter: Maximum multiplicative update iterations.
        cfg: Optional configuration object.
        target_layers: Optional layer name filter.
        max_error: Maximum relative Frobenius error threshold (default 0.35).
        fallback_to_original: If True, keep original layer on high reconstruction error.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 64,
        max_iter: int = 100,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        max_error: float = 0.35,
        fallback_to_original: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="nmf", **kwargs)
        self.rank = rank or getattr(cfg, "nmf_rank", None) or getattr(cfg, "rank", 64)
        self.max_iter = max_iter or getattr(cfg, "nmf_max_iter", 100)
        self.cfg = cfg
        self.target_layers = target_layers or getattr(cfg, "target_layers", None)
        self.max_error = float(getattr(cfg, "max_error", max_error))
        self.fallback_to_original = bool(getattr(cfg, "fallback_to_original", fallback_to_original))
        self.compressed_layers: List[Dict[str, Any]] = []

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if any(k in name.lower() for k in ["classifier", "score", "lm_head", "pooler", "embedding"]):
            return False
        cls_name = module.__class__.__name__
        is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name.lower()
        if not is_linear_like:
            return False
        if isinstance(module, NMFLinear):
            return False
        from framework.utils.compat import should_compress_linear
        if not should_compress_linear(self.model, name):
            return False
        if self.target_layers is None:
            return True
        return any(t in name for t in self.target_layers)

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress model linear layers via NMF."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        from framework.core.model_utils import replace_module

        self.compressed_layers = []
        for name, module in list(self.model.named_modules()):
            if self._is_target_module(name, module):
                raw_w = module.weight.data
                cls_name = module.__class__.__name__
                is_conv1d = "conv1d" in cls_name.lower() or (
                    raw_w.ndim == 2
                    and hasattr(module, "in_features")
                    and module.in_features == raw_w.shape[0]
                    and raw_w.shape[0] < raw_w.shape[1]
                )
                W_orig = raw_w.t().float() if is_conv1d else raw_w.float()
                max_rank = min(W_orig.shape[0], W_orig.shape[1])
                r = min(self.rank, max_rank)

                nmf_layer = NMFLinear.from_linear(module, rank=r, max_iter=self.max_iter)
                W_rec = nmf_layer.weight.float()
                err = (torch.norm(W_orig - W_rec, p="fro") / (torch.norm(W_orig, p="fro") + 1e-8)).item()

                if self.fallback_to_original and err > self.max_error:
                    continue

                replace_module(self.model, name, nmf_layer)
                self.compressed_layers.append({
                    "name": name,
                    "rank": nmf_layer.rank,
                    "relative_error": float(err),
                })

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layer_count": len(self.compressed_layers),
                "nmf_rank": self.rank,
                "max_iter": self.max_iter,
                "compressed_layers": self.compressed_layers,
            }
        )
        return self.model, stats

    def compress_linear(self, linear: nn.Module) -> NMFLinear:
        """Compress a single linear layer into NMFLinear."""
        return NMFLinear.from_linear(linear, rank=self.rank, max_iter=self.max_iter)


@COMPRESSOR_REGISTRY.register("nmf")
@COMPRESSOR_REGISTRY.register("nmf_compressor")
def nmf_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for NMF compression."""
    effective_rank = rank or getattr(comp_config, "nmf_rank", None) or getattr(comp_config, "rank", 64)
    max_iter = getattr(comp_config, "nmf_max_iter", 100)
    compressor = NMFCompressor(model, rank=effective_rank, max_iter=max_iter, cfg=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model

