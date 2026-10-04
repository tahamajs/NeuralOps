"""
GreCoDeC: Greedy Coordinate Descent Low-Rank Compressor.
Registered in COMPRESSOR_REGISTRY as 'grecodec'.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.pruning_utils import adaptive_max_error
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract config value from None, dict, or namespace/object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "c_attn",
    "c_proj",
    "query",
    "key",
    "value",
}


class GreCoDeCLinear(nn.Module):
    is_custom_attention: bool = True
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
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (lora_alpha / lora_r) if lora_r > 0 else 1.0

        requires_grad = not (self.lora_r > 0)
        self.U = nn.Parameter(
            torch.empty(out_features, self.rank, dtype=dtype, device=device),
            requires_grad=requires_grad,
        )
        self.V = nn.Parameter(
            torch.empty(self.rank, in_features, dtype=dtype, device=device),
            requires_grad=requires_grad,
        )

        bound = 1.0 / math.sqrt(max(1, in_features))
        nn.init.uniform_(self.U, -bound, bound)
        nn.init.uniform_(self.V, -bound, bound)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=dtype, device=device),
                requires_grad=requires_grad,
            )
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, in_features), dtype=torch.float32, device=device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_r), dtype=torch.float32, device=device)
            )
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

        out = out
        return out.view(*orig_shape[:-1], self.out_features)

    def get_weight(self) -> torch.Tensor:
        """Materialize full reconstructed weight matrix W = U @ V."""
        return torch.matmul(self.U, self.V)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear | Any,
        rank: int = 16,
        max_iter: int = 15,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> GreCoDeCLinear:
        """Construct GreCoDeCLinear by greedy coordinate descent low-rank factorization."""
        target_mod = getattr(linear, "base_layer", linear)
        cls_name = target_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        raw_w = target_mod.weight.data

        if is_conv1d:
            in_f = raw_w.shape[0]
            out_f = raw_w.shape[1]
            W_float = raw_w.t().contiguous().float()  # [out_features, in_features]
        else:
            out_f = raw_w.shape[0]
            in_f = raw_w.shape[1]
            W_float = raw_w.float()

        bias = (
            target_mod.bias.data
            if getattr(target_mod, "bias", None) is not None and target_mod.bias is not None
            else None
        )
        r = max(1, min(rank, out_f, in_f))

        # Check existing LoRA parameters
        existing_lora_A = getattr(linear, "lora_A", None)
        existing_lora_B = getattr(linear, "lora_B", None)

        # Coordinate descent low-rank approximation with SVD initialization fallback
        U_init, S_init, Vt_init = torch.linalg.svd(W_float, full_matrices=False)
        sqrt_S = torch.sqrt(torch.clamp(S_init[:r], min=1e-12))
        U_mat = (U_init[:, :r] * sqrt_S.unsqueeze(0)).clone()
        V_mat = (sqrt_S.unsqueeze(1) * Vt_init[:r, :]).clone()

        # Iterative coordinate descent refinement
        for _ in range(max_iter):
            # Update U given V: min_U ||W - U V||_F^2 => U = W V^T (V V^T)^{-1}
            VVt = torch.matmul(V_mat, V_mat.t().contiguous())
            VVt_reg = VVt + 1e-6 * torch.eye(r, device=VVt.device, dtype=VVt.dtype)
            try:
                WV_t = torch.matmul(W_float, V_mat.t().contiguous())
                U_mat = torch.linalg.solve(VVt_reg, WV_t.t().contiguous()).t().contiguous()
            except Exception:
                break

            # Update V given U: min_V ||W - U V||_F^2 => V = (U^T U)^{-1} U^T W
            UtU = torch.matmul(U_mat.t().contiguous(), U_mat)
            UtU_reg = UtU + 1e-6 * torch.eye(r, device=UtU.device, dtype=UtU.dtype)
            try:
                UtW = torch.matmul(U_mat.t().contiguous(), W_float)
                V_mat = torch.linalg.solve(UtU_reg, UtW)
            except Exception:
                break

        # Symmetrical norm balancing
        U_norm = torch.norm(U_mat, p=2, dim=0, keepdim=True).clamp_min(1e-8)
        V_norm = torch.norm(V_mat, p=2, dim=1, keepdim=True).clamp_min(1e-8)
        geom = torch.sqrt(U_norm.squeeze(0) * V_norm.squeeze(1))
        U_data = U_mat * (geom / U_norm.squeeze(0)).unsqueeze(0)
        V_data = V_mat * (geom / V_norm.squeeze(1)).unsqueeze(1)

        effective_lora_r = lora_r
        if existing_lora_A is not None and hasattr(existing_lora_A, "weight"):
            effective_lora_r = existing_lora_A.weight.shape[0]

        grecodec_layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=bias is not None,
            lora_r=effective_lora_r,
            lora_alpha=lora_alpha,
            dtype=raw_w.dtype,
            device=raw_w.device,
        )

        grecodec_layer.U.data.copy_(U_data.to(dtype=raw_w.dtype, device=raw_w.device))
        grecodec_layer.V.data.copy_(V_data.to(dtype=raw_w.dtype, device=raw_w.device))

        if bias is not None and grecodec_layer.bias is not None:
            grecodec_layer.bias.data.copy_(bias.to(dtype=raw_w.dtype, device=raw_w.device))

        if (
            existing_lora_A is not None
            and existing_lora_B is not None
            and grecodec_layer.lora_A is not None
        ):
            w_A = getattr(existing_lora_A, "weight", existing_lora_A)
            w_B = getattr(existing_lora_B, "weight", existing_lora_B)
            if hasattr(w_A, "data") and hasattr(w_B, "data"):
                grecodec_layer.lora_A.data.copy_(w_A.data)
                grecodec_layer.lora_B.data.copy_(w_B.data)
                grecodec_layer.lora_A.requires_grad = True
                grecodec_layer.lora_B.requires_grad = True

        return grecodec_layer


@COMPRESSOR_REGISTRY.register("grecodec_compressor_class")
class GreCoDeCCompressor(BaseCompressor):
    """
    GreCoDeC Compressor.
    Decomposes linear layers using coordinate descent low-rank factorization.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        rank: int = 16,
        max_iter: int = 15,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        max_error: float = 0.35,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        skip_attention_qkv: bool = False,
        cfg: Any | None = None,
        target_layers: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="grecodec", **kwargs)
        cfg = cfg or kwargs.get("comp_config")
        self.cfg = cfg

        self.rank = int(_cfg_get(cfg, "grecodec_rank", _cfg_get(cfg, "rank", rank)))
        self.max_iter = int(_cfg_get(cfg, "grecodec_max_iter", _cfg_get(cfg, "max_iter", max_iter)))
        self.lora_r = int(_cfg_get(cfg, "lora_r", _cfg_get(cfg, "lora_rank", lora_r)))
        self.lora_alpha = float(_cfg_get(cfg, "lora_alpha", lora_alpha))
        c_max_error = _cfg_get(cfg, "grecodec_max_error", _cfg_get(cfg, "max_error", max_error))
        if c_max_error is None or c_max_error == 0.35:
            c_max_error = adaptive_max_error(self.rank, 768, 768, "distilbert")

        self.max_error = float(c_max_error)
        self.target_attention_only = bool(
            _cfg_get(
                cfg,
                "grecodec_target_attention_only",
                _cfg_get(cfg, "target_attention_only", target_attention_only),
            )
        )
        self.fallback_to_original = bool(
            _cfg_get(
                cfg,
                "grecodec_fallback_to_original",
                _cfg_get(cfg, "fallback_to_original", fallback_to_original),
            )
        )
        self.skip_attention_qkv = bool(_cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv))
        self.target_layers = target_layers or _cfg_get(cfg, "target_layers", None)

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads:
            name_lower = name.lower()
            if any(
                p in name_lower
                for p in [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "query",
                    "key",
                    "value",
                    "c_attn",
                ]
            ):
                return True
        return False

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
            return False
        if any(
            skip in name_lower
            for skip in [
                "classifier",
                "score",
                "lm_head",
                "pooler",
                "embedding",
                "router",
                "gate",
            ]
        ):
            return False

        if self.target_attention_only:
            if not any(attn_kw in name_lower for attn_kw in _ATTN_NAMES):
                return False

        if self._is_gqa_asymmetric(name, module):
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        if not (is_linear or is_conv1d) or isinstance(target_mod, GreCoDeCLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_layers is not None:
            return any(t in name for t in self.target_layers)
        return True

    def compress(self, model: nn.Module | None = None) -> tuple[nn.Module, dict[str, Any]]:
        """Compress linear layers using GreCoDeC decomposition."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        seen_weights: set[int] = set()

        total_orig_params = 0
        total_comp_params = 0

        # Freeze skipped non-target layers if target_attention_only is enabled
        if self.target_attention_only:
            for name, module in self.model.named_modules():
                if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
                    continue
                target_mod = getattr(module, "base_layer", module)
                cls_name = target_mod.__class__.__name__.lower()
                is_linear = isinstance(target_mod, nn.Linear)
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                if (is_linear or is_conv1d) and not self._is_target_module(name, module):
                    for p in module.parameters():
                        p.requires_grad = False
                    self.skipped_layers.append(name)

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            target_mod = getattr(module, "base_layer", module)
            if not hasattr(target_mod, "weight") or target_mod.weight is None:
                continue

            w_id = id(target_mod.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            cls_name = target_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
            raw_w = target_mod.weight.data
            W_target = raw_w.t().contiguous().float() if is_conv1d else raw_w.float()
            out_f, in_f = W_target.shape[0], W_target.shape[1]
            orig_params = in_f * out_f
            total_orig_params += orig_params

            max_rank = min(in_f, out_f)
            bound_r = (in_f * out_f) // (in_f + out_f)
            if max_rank < 16 or bound_r < 4:
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                total_comp_params += orig_params
                continue

            try:
                grecodec_layer = GreCoDeCLinear.from_linear(
                    module,
                    rank=self.rank,
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
                        for p in module.parameters():
                            p.requires_grad = False
                        total_comp_params += orig_params
                        continue

                replace_module(self.model, name, grecodec_layer)
                self.compressed_layers.append(name)
                comp_params = grecodec_layer.rank * (in_f + out_f)
                if grecodec_layer.lora_r > 0:
                    comp_params += grecodec_layer.lora_r * (in_f + out_f)
                total_comp_params += comp_params
            except Exception as e:
                logger.debug("GreCoDeC error on %s: %s", name, e)
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                total_comp_params += orig_params

        # Post-compression forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                first_p = next(self.model.parameters(), None)
                dev = first_p.device if first_p is not None else torch.device("cpu")
                dummy = torch.randint(0, 100, (1, 8), device=dev)
                try:
                    self.model(dummy)
                except Exception:
                    first_linear = next(
                        (
                            m
                            for m in self.model.modules()
                            if isinstance(m, (nn.Linear, GreCoDeCLinear))
                        ),
                        None,
                    )
                    in_dim = getattr(first_linear, "in_features", 8)
                    self.model(torch.randn(1, 8, in_dim, device=dev))
        except Exception:
            forward_check_success = False

        effective_ratio = total_comp_params / total_orig_params if total_orig_params > 0 else 1.0

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (
            (sum(self.layer_errors.values()) / len(self.layer_errors)) if self.layer_errors else 0.0
        )
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "compressed_layer_count": len(self.compressed_layers),
                "fallback_layers": self.fallback_layers,
                "fallback_layer_count": len(self.fallback_layers),
                "skipped_layers": self.skipped_layers,
                "skipped_layer_count": len(self.skipped_layers),
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "num_skipped": len(self.skipped_layers),
                "layer_errors": self.layer_errors,
                "avg_reconstruction_error": avg_err,
                "grecodec_rank": self.rank,
                "grecodec_max_iter": self.max_iter,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "effective_parameter_ratio": round(effective_ratio, 4),
                "forward_check_success": forward_check_success,
            }
        )
        logger.info(
            "Compressor grecodec finished: %d compressed, %d fallback, avg_err=%.4f, max_err_threshold=%.4f",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            self.max_error,
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("grecodec")
@COMPRESSOR_REGISTRY.register("grecodec_compressor")
@COMPRESSOR_REGISTRY.register("grecodec_factory")
@COMPRESSOR_REGISTRY.register("grecodec_compressor_factory")
def grecodec_compressor_factory(
    model: nn.Module,
    comp_config: Any | None = None,
    rank: int | None = None,
    **kwargs: Any,
) -> nn.Module | tuple[nn.Module, dict[str, Any]]:
    """Factory function for GreCoDeC compressor."""
    effective_rank = rank or _cfg_get(
        comp_config, "grecodec_rank", _cfg_get(comp_config, "rank", 16)
    )
    max_iter = _cfg_get(comp_config, "grecodec_max_iter", _cfg_get(comp_config, "max_iter", 15))
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(comp_config, "grecodec_max_error", _cfg_get(comp_config, "max_error", 0.35)),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(
            comp_config,
            "grecodec_target_attention_only",
            _cfg_get(comp_config, "target_attention_only", False),
        ),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(
            comp_config,
            "grecodec_fallback_to_original",
            _cfg_get(comp_config, "fallback_to_original", True),
        ),
    )
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = GreCoDeCCompressor(
        model,
        rank=effective_rank,
        max_iter=max_iter,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_error=max_error,
        target_attention_only=target_attention_only,
        fallback_to_original=fallback,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    try:
        compressed_model.compression_stats = stats
    except Exception:
        pass
    if return_stats:
        return compressed_model, stats
    return compressed_model


grecodec_factory = grecodec_compressor_factory

__all__ = [
    "GreCoDeCLinear",
    "GreCoDeCCompressor",
    "grecodec_compressor_factory",
    "grecodec_factory",
]
