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
    Feature-Regression Aligned Factorized Linear Layer with optional LoRA adapter.
    W ≈ B @ A
    - A in R^{rank x in_features}: input feature-aligned basis
    - B in R^{out_features x rank}: output expansion matrix

    Forward pass:
    y = ((x @ A^T) @ B^T) + bias + (x @ lora_A^T @ lora_B^T) * (alpha / r)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        bias: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (lora_alpha / lora_r) if lora_r > 0 else 1.0

        requires_grad = not (self.lora_r > 0)
        A_init = torch.empty(self.rank, in_features, dtype=torch.float32, device=device)
        nn.init.orthogonal_(A_init)
        self.A = nn.Parameter(A_init.to(dtype=dtype), requires_grad=requires_grad)
        self.B = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device), requires_grad=requires_grad)
        nn.init.xavier_uniform_(self.B)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device), requires_grad=requires_grad)
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(torch.empty((self.lora_r, in_features), dtype=torch.float32, device=device))
            self.lora_B = nn.Parameter(torch.zeros((out_features, self.lora_r), dtype=torch.float32, device=device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

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

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + self.scaling * F.linear(
                F.linear(x_flat.float(), self.lora_A), self.lora_B
            ).to(out.dtype)

        return out.view(*orig_shape[:-1], self.out_features)

    def get_weight(self) -> torch.Tensor:
        """Materialize full weight matrix."""
        return torch.matmul(self.B, self.A)

    @classmethod
    def from_linear(
        cls,
        linear: Union[nn.Linear, Any],
        rank: int = 8,
        energy_threshold: Optional[float] = None,
        cov_matrix: Optional[torch.Tensor] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> "FRanDILinear":
        """
        Construct FRanDILinear with feature-aligned SVD or energy threshold selection.
        Supports nn.Linear and Conv1D.
        """
        cls_name = linear.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(linear, "nf")
        raw_w = linear.weight.data

        if is_conv1d:
            W = raw_w.t().float()
        else:
            W = raw_w.float()

        bias = linear.bias.data if getattr(linear, "bias", None) is not None and linear.bias is not None else None
        out_f, in_f = W.shape
        W_float = W.clone()

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
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            dtype=raw_w.dtype,
            device=raw_w.device,
        )

        frandi_layer.A.data.copy_(A_data[:r, :].to(dtype=raw_w.dtype, device=raw_w.device))
        frandi_layer.B.data.copy_(B_data[:, :r].to(dtype=raw_w.dtype, device=raw_w.device))

        if bias is not None and frandi_layer.bias is not None:
            frandi_layer.bias.data.copy_(bias.to(dtype=raw_w.dtype, device=raw_w.device))

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
        lora_r: Optional LoRA rank for trainable adapters.
        lora_alpha: LoRA scaling alpha.
        max_error: Maximum reconstruction error before fallback.
        fallback_to_original: If True, keep original layer on error excess.
        skip_attention_qkv: Skip asymmetric GQA projections.
        cfg: Optional configuration object.
        target_layers: Optional target layer filter.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 8,
        energy_threshold: Optional[float] = None,
        calibration_loader: Optional[Any] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        max_error: float = 0.25,
        fallback_to_original: bool = True,
        skip_attention_qkv: bool = True,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="frandi", **kwargs)
        self.rank = int(rank or getattr(cfg, "frandi_rank", None) or getattr(cfg, "rank", 8))
        self.energy_threshold = (
            energy_threshold
            if energy_threshold is not None
            else getattr(cfg, "frandi_energy", None) or getattr(cfg, "energy_threshold", None)
        )
        self.calibration_loader = calibration_loader or getattr(cfg, "calibration_loader", None)
        self.lora_r = int(getattr(cfg, "lora_r", getattr(cfg, "lora_rank", lora_r)))
        self.lora_alpha = float(getattr(cfg, "lora_alpha", lora_alpha))
        self.max_error = float(getattr(cfg, "max_error", max_error))
        self.fallback_to_original = bool(getattr(cfg, "fallback_to_original", fallback_to_original))
        self.skip_attention_qkv = bool(getattr(cfg, "skip_attention_qkv", skip_attention_qkv))
        self.cfg = cfg
        self.target_layers = target_layers or getattr(cfg, "target_layers", None)
        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads:
            name_lower = name.lower()
            if any(p in name_lower for p in ["q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn"]):
                return True
        return False

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
                    cls_name = module.__class__.__name__.lower()
                    is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
                    in_f = module.weight.shape[0] if is_conv1d else module.weight.shape[1]
                    x_flat = x.reshape(-1, in_f)
                    cov = torch.matmul(x_flat.t(), x_flat)
                    if name not in covariances:
                        covariances[name] = cov
                        counts[name] = x_flat.size(0)
                    else:
                        covariances[name] += cov
                        counts[name] += x_flat.size(0)
            return hook

        for name, module in self.model.named_modules():
            cls_name = module.__class__.__name__.lower()
            is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
            if is_linear_like and not isinstance(module, FRanDILinear) and should_compress_linear(self.model, name):
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
        from framework.core.model_utils import replace_module

        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        covs = self._collect_covariances()
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        seen_weights: set[int] = set()

        for name, module in list(self.model.named_modules()):
            cls_name = module.__class__.__name__.lower()
            is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
            if not is_linear_like or isinstance(module, FRanDILinear):
                continue

            if not should_compress_linear(self.model, name):
                continue

            if self._is_gqa_asymmetric(name, module):
                continue

            if self.target_layers and not any(t in name for t in self.target_layers):
                continue

            if not hasattr(module, "weight") or module.weight is None:
                continue

            w_id = id(module.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            raw_w = module.weight.data
            is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
            W_target = raw_w.t().float() if is_conv1d else raw_w.float()
            out_f, in_f = W_target.shape[0], W_target.shape[1]
            max_rank = min(in_f, out_f)
            if max_rank < 16:
                continue

            r = min(self.rank, max_rank)
            if r >= max_rank and self.energy_threshold is None and max_rank > 4:
                continue

            cov = covs.get(name, None)
            try:
                frandi_layer = FRanDILinear.from_linear(
                    module,
                    rank=self.rank,
                    energy_threshold=self.energy_threshold,
                    cov_matrix=cov,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )
                W_comp = frandi_layer.get_weight().float()
                W_norm = torch.norm(W_target, p="fro").clamp_min(1e-8)
                err = float(torch.norm(W_target - W_comp, p="fro").item() / W_norm)
                self.layer_errors[name] = err

                if err > self.max_error:
                    if self.fallback_to_original:
                        self.fallback_layers.append(name)
                        continue

                replace_module(self.model, name, frandi_layer)
                self.compressed_layers.append(name)
            except Exception as e:
                self.fallback_layers.append(name)

        # Post-compression forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev)
                try:
                    self.model(dummy)
                except Exception:
                    first_linear = next(
                        (m for m in self.model.modules() if isinstance(m, (nn.Linear, FRanDILinear))),
                        None,
                    )
                    in_dim = getattr(first_linear, "in_features", 8)
                    self.model(torch.randn(1, 8, in_dim, device=dev))
        except Exception as e:
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (sum(self.layer_errors.values()) / len(self.layer_errors)) if self.layer_errors else 0.0
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "compressed_layer_count": len(self.compressed_layers),
                "fallback_layers": self.fallback_layers,
                "fallback_layer_count": len(self.fallback_layers),
                "skipped_layers": self.skipped_layers,
                "skipped_layer_count": len(self.skipped_layers),
                "layer_errors": self.layer_errors,
                "avg_reconstruction_error": avg_err,
                "frandi_rank": self.rank,
                "energy_threshold": self.energy_threshold,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "forward_check_success": forward_check_success,
            }
        )
        return self.model, stats

    def compress_linear(
        self, linear: nn.Module, cov_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[FRanDILinear, float]:
        """Compress a single linear layer into FRanDILinear."""
        cls_name = linear.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(linear, "nf")
        raw_w = linear.weight.data
        W_target = raw_w.t().float() if is_conv1d else raw_w.float()
        layer = FRanDILinear.from_linear(
            linear,
            rank=self.rank,
            energy_threshold=self.energy_threshold,
            cov_matrix=cov_matrix,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
        )
        W_comp = layer.get_weight().float()
        W_norm = torch.norm(W_target, p="fro").clamp_min(1e-8)
        err = float(torch.norm(W_target - W_comp, p="fro").item() / W_norm)
        return layer, err


@COMPRESSOR_REGISTRY.register("frandi_factory")
@COMPRESSOR_REGISTRY.register("frandi_compressor_factory")
def frandi_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for FRanDI compressor."""
    effective_rank = rank or getattr(comp_config, "frandi_rank", None) or getattr(comp_config, "rank", 8)
    energy = getattr(comp_config, "frandi_energy", None) or getattr(comp_config, "energy_threshold", None)
    lora_r = kwargs.pop("lora_r", getattr(comp_config, "lora_r", 0) if comp_config else 0)
    lora_alpha = kwargs.pop("lora_alpha", getattr(comp_config, "lora_alpha", 1.0) if comp_config else 1.0)
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = FRanDICompressor(
        model,
        rank=effective_rank,
        energy_threshold=energy,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model
