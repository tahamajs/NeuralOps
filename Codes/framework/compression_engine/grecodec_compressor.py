"""
GreCoDeC: Greedy Coordinate Descent Low-Rank Compressor.
Registered in COMPRESSOR_REGISTRY as 'grecodec'.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear


class GreCoDeCLinear(nn.Module):
    """
    Greedy Coordinate Descent Factorized Linear Layer with optional LoRA adapter.
    W ≈ U @ V
    - U in R^{out_features x rank}
    - V in R^{rank x in_features}

    Forward pass:
    y = ((x @ V^T) @ U^T) + bias + (x @ lora_A^T @ lora_B^T) * (alpha / r)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
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
        self.U = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device), requires_grad=requires_grad)
        self.V = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device), requires_grad=requires_grad)

        bound = 1.0 / math.sqrt(max(1, in_features))
        nn.init.uniform_(self.U, -bound, bound)
        nn.init.uniform_(self.V, -bound, bound)

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
        return torch.matmul(self.U, self.V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sub-quadratic forward execution.
        y = ((x @ V^T) @ U^T) + bias + LoRA
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        x_v = F.linear(x_flat, self.V)
        out = F.linear(x_v, self.U, self.bias)

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + self.scaling * F.linear(
                F.linear(x_flat.float(), self.lora_A), self.lora_B
            ).to(out.dtype)

        return out.view(*orig_shape[:-1], self.out_features)

    def get_weight(self) -> torch.Tensor:
        """Materialize full reconstructed weight matrix W = U @ V."""
        return torch.matmul(self.U, self.V)

    @classmethod
    def from_linear(
        cls,
        linear: Union[nn.Linear, Any],
        rank: int = 16,
        max_iter: int = 15,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> "GreCoDeCLinear":
        """
        Decompose linear layer weight matrix via Greedy Coordinate Descent (GreCoDeC)
        with ALS iterations and Gram-Schmidt re-orthogonalization, with SVD fallback.
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
        r = max(1, min(rank, out_f, in_f))

        grecodec_layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=bias is not None,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            dtype=raw_w.dtype,
            device=raw_w.device,
        )

        R = W.clone()
        U_factors = torch.zeros(out_f, r, dtype=torch.float32, device=raw_w.device)
        V_factors = torch.zeros(r, in_f, dtype=torch.float32, device=raw_w.device)

        success = True
        try:
            for k in range(r):
                # 1. Initialize v_k from row of R with highest residual norm
                row_norms = (R ** 2).sum(dim=1)
                best_row = torch.argmax(row_norms).item()
                v_k = R[best_row, :].clone()
                v_norm = v_k.norm().clamp_min(1e-8)
                v_k = (v_k / v_norm).unsqueeze(1)  # (in_f, 1)

                # 2. ALS coordinate descent with Gram-Schmidt re-orthogonalization
                u_k = torch.zeros(out_f, 1, dtype=torch.float32, device=raw_w.device)
                for _ in range(max(1, max_iter)):
                    # u = R @ v / ||v||^2
                    v_sq = (v_k ** 2).sum().clamp_min(1e-10)
                    u_k = torch.matmul(R, v_k) / v_sq

                    # Gram-Schmidt re-orthogonalization on u_k against prior columns U_factors[:, :k]
                    if k > 0:
                        prior_U = U_factors[:, :k]  # (out_f, k)
                        u_proj = torch.matmul(prior_U, torch.matmul(prior_U.t(), u_k) / (prior_U.norm(dim=0, keepdim=True) ** 2).clamp_min(1e-10).t())
                        u_k = u_k - u_proj

                    u_sq = (u_k ** 2).sum().clamp_min(1e-10)
                    # v = R^T @ u / ||u||^2
                    v_k = torch.matmul(R.t(), u_k) / u_sq

                    # Gram-Schmidt re-orthogonalization on v_k against prior rows V_factors[:k, :]
                    if k > 0:
                        prior_V = V_factors[:k, :].t()  # (in_f, k)
                        v_proj = torch.matmul(prior_V, torch.matmul(prior_V.t(), v_k) / (prior_V.norm(dim=0, keepdim=True) ** 2).clamp_min(1e-10).t())
                        v_k = v_k - v_proj

                # Check for degenerate vector
                if u_k.norm() < 1e-8 or v_k.norm() < 1e-8 or torch.isnan(u_k).any() or torch.isnan(v_k).any():
                    success = False
                    break

                # 3. Deflate residual: R = R - u @ v^T
                R = R - torch.matmul(u_k, v_k.t())

                U_factors[:, k] = u_k.squeeze(1)
                V_factors[k, :] = v_k.squeeze(1)

        except Exception:
            success = False

        if not success or torch.isnan(U_factors).any() or torch.isnan(V_factors).any():
            # Robust SVD fallback to guarantee orthogonal singular components
            U_svd, S_svd, Vh_svd = torch.linalg.svd(W, full_matrices=False)
            sqrt_S = torch.sqrt(torch.clamp(S_svd[:r], min=1e-12))
            U_factors = U_svd[:, :r] * sqrt_S.unsqueeze(0)
            V_factors = sqrt_S.unsqueeze(1) * Vh_svd[:r, :]

        grecodec_layer.U.data.copy_(U_factors.to(dtype=raw_w.dtype))
        grecodec_layer.V.data.copy_(V_factors.to(dtype=raw_w.dtype))

        if bias is not None and grecodec_layer.bias is not None:
            grecodec_layer.bias.data.copy_(bias)

        return grecodec_layer


@COMPRESSOR_REGISTRY.register("grecodec")
@COMPRESSOR_REGISTRY.register("grecodec_compressor")
class GreCoDeCCompressor(BaseCompressor):
    """
    GreCoDeC (Greedy Coordinate Descent) Compressor.
    Decomposes linear/Conv1D layers into low-rank factors using fast greedy rank-1 updates.

    Args:
        model: PyTorch model.
        rank: Target rank for decomposition.
        max_iter: Coordinate descent iterations per rank component (default 15).
        lora_r: Optional LoRA rank.
        lora_alpha: LoRA scaling alpha.
        max_error: Maximum allowed reconstruction error before fallback (default 0.20).
        fallback_to_original: If True, keep original layer on error excess.
        skip_attention_qkv: Skip asymmetric GQA projections.
        cfg: Optional configuration object.
        target_layers: Optional target layer filter.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 16,
        max_iter: int = 15,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        skip_attention_qkv: bool = True,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="grecodec", **kwargs)
        self.rank = int(rank or getattr(cfg, "grecodec_rank", None) or getattr(cfg, "rank", 16))
        self.max_iter = int(max_iter or getattr(cfg, "grecodec_max_iter", 15))
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

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear layers using GreCoDeC."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        seen_weights: set[int] = set()

        for name, module in list(self.model.named_modules()):
            cls_name = module.__class__.__name__.lower()
            is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name or hasattr(module, "nf")
            if not is_linear_like or isinstance(module, GreCoDeCLinear):
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
            r = max(1, min(self.rank, max_rank))

            if r >= max_rank and max_rank > 4:
                continue

            try:
                grecodec_layer = GreCoDeCLinear.from_linear(
                    module,
                    rank=r,
                    max_iter=self.max_iter,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )
                W_comp = grecodec_layer.get_weight().float()
                W_norm = torch.norm(W_target, p="fro").clamp_min(1e-8)
                err = float(torch.norm(W_target - W_comp, p="fro").item() / W_norm)
                self.layer_errors[name] = err

                if err > self.max_error:
                    if self.fallback_to_original:
                        self.fallback_layers.append(name)
                        continue

                replace_module(self.model, name, grecodec_layer)
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
                        (m for m in self.model.modules() if isinstance(m, (nn.Linear, GreCoDeCLinear))),
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
                "grecodec_rank": self.rank,
                "grecodec_max_iter": self.max_iter,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "forward_check_success": forward_check_success,
            }
        )
        return self.model, stats

    def compress_linear(self, linear: nn.Module) -> Tuple[GreCoDeCLinear, float]:
        """Compress a single linear layer into GreCoDeCLinear."""
        cls_name = linear.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(linear, "nf")
        raw_w = linear.weight.data
        W_target = raw_w.t().float() if is_conv1d else raw_w.float()
        layer = GreCoDeCLinear.from_linear(
            linear,
            rank=self.rank,
            max_iter=self.max_iter,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
        )
        W_comp = layer.get_weight().float()
        W_norm = torch.norm(W_target, p="fro").clamp_min(1e-8)
        err = float(torch.norm(W_target - W_comp, p="fro").item() / W_norm)
        return layer, err


@COMPRESSOR_REGISTRY.register("grecodec_factory")
@COMPRESSOR_REGISTRY.register("grecodec_compressor_factory")
def grecodec_compressor_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for GreCoDeC compressor."""
    effective_rank = rank or getattr(comp_config, "grecodec_rank", None) or getattr(comp_config, "rank", 16)
    max_iter = getattr(comp_config, "grecodec_max_iter", 15)
    lora_r = kwargs.pop("lora_r", getattr(comp_config, "lora_r", 0) if comp_config else 0)
    lora_alpha = kwargs.pop("lora_alpha", getattr(comp_config, "lora_alpha", 1.0) if comp_config else 1.0)
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = GreCoDeCCompressor(
        model,
        rank=effective_rank,
        max_iter=max_iter,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model
