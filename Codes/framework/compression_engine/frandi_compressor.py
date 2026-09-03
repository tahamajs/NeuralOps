"""
FRanDI: Feature-Regression Aligned Singular Value Decomposition Compressor.
Registered in COMPRESSOR_REGISTRY as 'frandi'.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear


class FRanDILinear(nn.Module):
    """
    Feature-Regression Aligned Factorized Linear Layer.
    W ≈ B @ A
    - A in R^{rank x in_features}: input feature-aligned basis
    - B in R^{out_features x rank}: output expansion matrix

    Forward pass:
    y = ((x @ A^T) @ B^T) + bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)

        A_init = torch.empty(self.rank, in_features, dtype=torch.float32, device=device)
        nn.init.orthogonal_(A_init)
        self.A = nn.Parameter(A_init.to(dtype=dtype))
        self.B = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))
        nn.init.xavier_uniform_(self.B)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        return torch.matmul(self.B, self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        x_a = F.linear(x_flat, self.A)
        out = F.linear(x_a, self.B, self.bias)

        return out.view(*orig_shape[:-1], self.out_features)

    def get_weight(self) -> torch.Tensor:
        """Materialize full weight matrix."""
        return torch.matmul(self.B, self.A)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 8,
        energy_threshold: Optional[float] = None,
        cov_matrix: Optional[torch.Tensor] = None,
    ) -> "FRanDILinear":
        """
        Construct FRanDILinear with feature-aligned SVD or energy threshold selection.
        """
        W = linear.weight.data
        bias = linear.bias.data if linear.bias is not None else None
        out_f, in_f = W.shape
        W_float = W.float()

        if cov_matrix is not None and cov_matrix.shape == (in_f, in_f):
            try:
                # Sqrt of covariance with eigenvalue clamping lambda_i >= 1e-4
                L, V = torch.linalg.eigh(cov_matrix.float() + 1e-4 * torch.eye(in_f, device=cov_matrix.device))
                L_clamped = torch.clamp(L, min=1e-4)
                L_sqrt = torch.diag(torch.sqrt(L_clamped))
                L_inv_sqrt = torch.diag(1.0 / torch.sqrt(L_clamped))
                C_sqrt = torch.matmul(torch.matmul(V, L_sqrt), V.t())
                C_inv_sqrt = torch.matmul(torch.matmul(V, L_inv_sqrt), V.t())

                W_scaled = torch.matmul(W_float, C_sqrt)
                U, S, Vt = torch.linalg.svd(W_scaled, full_matrices=False)

                r = cls._select_rank(S, rank=rank, energy_threshold=energy_threshold)
                r = max(1, min(r, out_f, in_f))
                sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                B_data = U[:, :r] * sqrt_S.unsqueeze(0)
                A_scaled = sqrt_S.unsqueeze(1) * Vt[:r, :]
                A_data = torch.matmul(A_scaled, C_inv_sqrt)

                # Symmetrically normalize factor magnitudes between A and B
                norm_B = torch.norm(B_data, p="fro").clamp_min(1e-8)
                norm_A = torch.norm(A_data, p="fro").clamp_min(1e-8)
                scale_factor = torch.sqrt(norm_B / norm_A)
                B_data = B_data / scale_factor
                A_data = A_data * scale_factor
            except Exception:
                U, S, Vt = torch.linalg.svd(W_float, full_matrices=False)
                r = cls._select_rank(S, rank=rank, energy_threshold=energy_threshold)
                r = max(1, min(r, out_f, in_f))
                sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                B_data = U[:, :r] * sqrt_S.unsqueeze(0)
                A_data = sqrt_S.unsqueeze(1) * Vt[:r, :]
        else:
            U, S, Vt = torch.linalg.svd(W_float, full_matrices=False)
            r = cls._select_rank(S, rank=rank, energy_threshold=energy_threshold)
            r = max(1, min(r, out_f, in_f))
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            B_data = U[:, :r] * sqrt_S.unsqueeze(0)
            A_data = sqrt_S.unsqueeze(1) * Vt[:r, :]
        frandi_layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=bias is not None,
            dtype=W.dtype,
            device=W.device,
        )

        frandi_layer.A.data.copy_(A_data[:r, :].to(dtype=W.dtype, device=W.device))
        frandi_layer.B.data.copy_(B_data[:, :r].to(dtype=W.dtype, device=W.device))

        if bias is not None and frandi_layer.bias is not None:
            frandi_layer.bias.data.copy_(bias.to(dtype=W.dtype, device=W.device))

        return frandi_layer

    @staticmethod
    def _select_rank(
        S: torch.Tensor, rank: int = 8, energy_threshold: Optional[float] = None
    ) -> int:
        if energy_threshold is None:
            return min(rank, len(S))
        total_energy = (S ** 2).sum().clamp_min(1e-12)
        cum_energy = (S ** 2).cumsum(dim=0) / total_energy
        idx = torch.where(cum_energy >= energy_threshold)[0]
        if len(idx) > 0:
            return idx[0].item() + 1
        return len(S)


@COMPRESSOR_REGISTRY.register("frandi")
@COMPRESSOR_REGISTRY.register("frandi_compressor")
class FRanDICompressor(BaseCompressor):
    """
    FRanDI (Feature-Regression aligned SVD) Compressor.
    Compresses linear layers using singular value decomposition with optional
    activation alignment and adaptive energy-preservation thresholding.

    Args:
        model: Model to compress.
        rank: Target rank when energy_threshold is None.
        energy_threshold: Energy ratio threshold (0 < energy_threshold <= 1.0).
        calibration_loader: Optional DataLoader for calibration statistics.
        cfg: Optional configuration object.
        target_layers: Optional target layer filter.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 8,
        energy_threshold: Optional[float] = None,
        calibration_loader: Optional[Any] = None,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="frandi", **kwargs)
        self.rank = rank or getattr(cfg, "frandi_rank", None) or getattr(cfg, "rank", 8)
        self.energy_threshold = (
            energy_threshold
            if energy_threshold is not None
            else getattr(cfg, "frandi_energy", None) or getattr(cfg, "energy_threshold", None)
        )
        self.calibration_loader = calibration_loader or getattr(cfg, "calibration_loader", None)
        self.cfg = cfg
        self.target_layers = target_layers or getattr(cfg, "target_layers", None)
        self.compressed_layers: List[str] = []

    def _collect_covariances(self) -> Dict[str, torch.Tensor]:
        """Compute empirical feature covariances from calibration loader."""
        if self.calibration_loader is None or not hasattr(self.model, "named_modules"):
            return {}

        covariances: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module, input, output):
                if len(input) > 0 and isinstance(input[0], torch.Tensor):
                    x = input[0].detach().float()
                    x_flat = x.reshape(-1, x.shape[-1])
                    cov = torch.matmul(x_flat.t(), x_flat)
                    if name not in covariances:
                        covariances[name] = cov
                        counts[name] = x_flat.size(0)
                    else:
                        covariances[name] += cov
                        counts[name] += x_flat.size(0)
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and not isinstance(module, FRanDILinear) and should_compress_linear(self.model, name):
                hooks.append(module.register_forward_hook(make_hook(name)))

        device = next(self.model.parameters()).device if list(self.model.parameters()) else torch.device("cpu")
        self.model.eval()
        with torch.no_grad():
            for batch in self.calibration_loader:
                if isinstance(batch, dict):
                    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    _ = self.model(**batch)
                elif isinstance(batch, (list, tuple)):
                    b0 = batch[0].to(device)
                    _ = self.model(b0)
                elif isinstance(batch, torch.Tensor):
                    _ = self.model(batch.to(device))

        for h in hooks:
            h.remove()

        for name in list(covariances.keys()):
            covariances[name] = covariances[name] / max(1, counts[name])

        return covariances

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear layers using FRanDI decomposition."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        covs = self._collect_covariances()
        self.compressed_layers = []

        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and not isinstance(module, FRanDILinear):
                name_lower = name.lower()
                if any(skip in name_lower for skip in ["lm_head", "classifier", "head", "output_layer", "score"]):
                    continue
                if self.target_layers and not any(t in name for t in self.target_layers):
                    continue

                max_rank = min(module.in_features, module.out_features)
                if max_rank < 16:
                    continue
                r = min(self.rank, max_rank)
                if r >= max_rank and self.energy_threshold is None and max_rank > 4:
                    continue

                cov = covs.get(name, None)
                frandi_layer = FRanDILinear.from_linear(
                    module,
                    rank=self.rank,
                    energy_threshold=self.energy_threshold,
                    cov_matrix=cov,
                )
                parent = self._get_parent_module(name)
                if parent is not None:
                    child_name = name.split(".")[-1]
                    setattr(parent, child_name, frandi_layer)
                    self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layer_count": len(self.compressed_layers),
                "frandi_rank": self.rank,
                "energy_threshold": self.energy_threshold,
            }
        )
        return self.model, stats

    def compress_linear(
        self, linear: nn.Linear, cov_matrix: Optional[torch.Tensor] = None
    ) -> FRanDILinear:
        """Compress a single linear layer into FRanDILinear."""
        return FRanDILinear.from_linear(
            linear,
            rank=self.rank,
            energy_threshold=self.energy_threshold,
            cov_matrix=cov_matrix,
        )

    def _get_parent_module(self, full_name: str) -> Optional[nn.Module]:
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None
        return parent


@COMPRESSOR_REGISTRY.register("frandi")
def frandi_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for FRanDI compressor."""
    effective_rank = rank or getattr(comp_config, "frandi_rank", None) or getattr(comp_config, "rank", 8)
    energy = getattr(comp_config, "frandi_energy", None) or getattr(comp_config, "energy_threshold", None)
    compressor = FRanDICompressor(
        model, rank=effective_rank, energy_threshold=energy, cfg=comp_config, **kwargs
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
