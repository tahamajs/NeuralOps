"""
SVD-based low-rank linear layer replacement with optional Fused SVD kernel acceleration.
Registered as 'svd_linear' and 'svd_linear_compressor'.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from neuralops.attentions.fused_svd_kernel import FusedSVDLinear, fused_svd_linear
except Exception:
    FusedSVDLinear = None
    fused_svd_linear = None

from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class SVDLinearLayer(nn.Module):
    is_custom_attention: bool = True
    """
    Replace a linear layer with factorized U @ V, keeping top-`rank` components.
    Optionally dispatches fused in-SRAM kernel to eliminate memory roundtrips.
    Includes optional trainable LoRA adapters (lora_A, lora_B).
    """

    def __init__(
        self,
        linear_layer: nn.Module,
        rank: int,
        use_fused_kernel: bool = False,
        use_rms_norm: bool = False,
        eps: float = 1e-6,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        base_layer = getattr(linear_layer, "base_layer", linear_layer)
        raw_w = base_layer.weight.data
        cls_name = base_layer.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(base_layer, "nf")

        if is_conv1d:
            W = raw_w.t().contiguous().float()
            self.out_features = W.shape[0]
            self.in_features = W.shape[1]
        else:
            W = raw_w.float()
            self.out_features = getattr(base_layer, "out_features", W.shape[0])
            self.in_features = getattr(base_layer, "in_features", W.shape[1])

        self.nf = self.out_features  # Conv1D compatibility
        param_cap = max(
            1,
            (self.in_features * self.out_features) // (self.in_features + self.out_features) - 1,
        )
        self.rank: int = max(1, min(rank, self.in_features, self.out_features, param_cap))
        self.use_fused_kernel: bool = use_fused_kernel
        self.use_rms_norm: bool = use_rms_norm
        self.eps: float = eps

        has_existing_lora = hasattr(linear_layer, "lora_A") and hasattr(linear_layer, "lora_B")
        effective_lora_r = lora_r
        if has_existing_lora and getattr(linear_layer, "r", 0) > 0:
            effective_lora_r = getattr(linear_layer, "r", lora_r)

        self.lora_r = int(effective_lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            sqrt_s = torch.sqrt(torch.clamp(S[: self.rank], min=1e-12))
            U_weighted = (U[:, : self.rank] * sqrt_s.unsqueeze(0)).to(device=device, dtype=dtype)
            V_weighted = (sqrt_s.unsqueeze(1) * Vh[: self.rank, :]).to(device=device, dtype=dtype)

        requires_grad = not (self.lora_r > 0)
        self.U = nn.Parameter(U_weighted, requires_grad=requires_grad)
        self.V = nn.Parameter(V_weighted, requires_grad=requires_grad)

        if use_rms_norm:
            self.norm_weight = nn.Parameter(torch.ones(self.rank, dtype=dtype, device=device))
        else:
            self.register_parameter("norm_weight", None)

        if base_layer.bias is not None:
            self.bias = nn.Parameter(base_layer.bias.data.clone(), requires_grad=requires_grad)
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, self.in_features), dtype=torch.float32, device=device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((self.out_features, self.lora_r), dtype=torch.float32, device=device)
            )
            if has_existing_lora:
                try:
                    if hasattr(linear_layer.lora_A, "weight"):
                        self.lora_A.data.copy_(linear_layer.lora_A.weight.data)
                    if hasattr(linear_layer.lora_B, "weight"):
                        self.lora_B.data.copy_(linear_layer.lora_B.weight.data)
                except Exception:
                    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            else:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        # Frobenius reconstruction error
        W_rec = torch.matmul(U_weighted.float(), V_weighted.float())
        norm_orig = torch.norm(W, p="fro")
        self.reconstruction_error = (torch.norm(W - W_rec, p="fro") / (norm_orig + 1e-8)).item()

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        return torch.matmul(self.U, self.V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional fused kernel acceleration and LoRA branch."""
        if self.use_fused_kernel and self.lora_r <= 0 and fused_svd_linear is not None:
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

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            orig_shape = x.shape
            x_flat = x.reshape(-1, self.in_features)
            lora_out = F.linear(F.linear(x_flat.float(), self.lora_A), self.lora_B)
            out = out + (self.scaling * lora_out).reshape(*orig_shape[:-1], self.out_features).to(
                out.dtype
            )

        return out


@COMPRESSOR_REGISTRY.register("svd_linear")
@COMPRESSOR_REGISTRY.register("svd_linear_compressor")
class SVDLinearCompressor(BaseCompressor):
    """Compresses model by replacing Linear/Conv1D layers with SVDLinearLayer."""

    def __init__(
        self,
        model: nn.Module,
        rank: int = 192,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        protect_critical: bool = True,
        target_attention_only: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = False,
        target_modules: list[str] | None = None,
        cfg: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="svd_linear", **kwargs)
        conf = cfg or kwargs.get("comp_config")
        if conf is not None:
            rank = _cfg_get(conf, "svd_rank", _cfg_get(conf, "rank", rank))
            max_error = _cfg_get(conf, "max_error", max_error)
            fallback_to_original = _cfg_get(conf, "fallback_to_original", fallback_to_original)
            protect_critical = _cfg_get(conf, "protect_critical", protect_critical)
            target_attention_only = _cfg_get(conf, "target_attention_only", target_attention_only)
            lora_r = _cfg_get(conf, "lora_r", _cfg_get(conf, "lora_rank", lora_r))
            lora_alpha = _cfg_get(conf, "lora_alpha", lora_alpha)
            skip_attention_qkv = _cfg_get(conf, "skip_attention_qkv", skip_attention_qkv)

        self.rank = int(rank)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.protect_critical = bool(protect_critical)
        self.target_attention_only = bool(target_attention_only)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.target_modules = target_modules
        self.layer_errors: dict[str, float] = {}
        self.replaced_layers: list[dict[str, Any]] = []
        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.stats: dict[str, Any] = {}

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
        name_l = name.lower()
        if any(
            k in name_l
            for k in [
                "out_lin",
                "out_proj",
                "o_proj",
                "c_proj",
                "output.dense",
                "self.output",
            ]
        ):
            return False
        if self.protect_critical and any(
            k in name_l
            for k in [
                "classifier",
                "score",
                "lm_head",
                "pooler",
                "embed",
                "logits_processor",
            ]
        ):
            return False

        if self.target_attention_only:
            attn_keywords = [
                "attn",
                "attention",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "c_attn",
                "c_proj",
                "query",
                "key",
                "value",
            ]
            if not any(k in name_l for k in attn_keywords):
                return False

        base_mod = getattr(module, "base_layer", module)
        cls_name = base_mod.__class__.__name__.lower()
        is_linear_like = (
            isinstance(base_mod, nn.Linear) or "conv1d" in cls_name or hasattr(base_mod, "nf")
        )
        if not is_linear_like:
            return False
        if isinstance(base_mod, SVDLinearLayer):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self._is_gqa_asymmetric(name, module):
            return False
        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        """Compress linear layers with SVD factorization."""
        self.layer_errors = {}
        self.replaced_layers = []
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        seen_weights: set[int] = set()

        target_layers = [
            (name, mod)
            for name, mod in list(self.model.named_modules())
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("SVDLinear: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        for name, module in target_layers:
            base_mod = getattr(module, "base_layer", module)
            if not hasattr(base_mod, "weight") or base_mod.weight is None:
                continue
            w_id = id(base_mod.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            try:
                new_layer = SVDLinearLayer(
                    module,
                    rank=self.rank,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )
                err = new_layer.reconstruction_error
                self.layer_errors[name] = err

                if err > self.max_error:
                    logger.warning(
                        "SVDLinear reconstruction error %.4f on %s exceeded threshold %.4f.",
                        err,
                        name,
                        self.max_error,
                    )
                    if self.fallback_to_original:
                        logger.info(
                            "Skipping SVDLinear compression for %s (fallback to original).",
                            name,
                        )
                        self.fallback_layers.append(name)
                        for p in module.parameters():
                            p.requires_grad = False
                        continue

                self.replaced_layers.append({"name": name, "rank": new_layer.rank, "error": err})
                self.compressed_layers.append(name)
                replace_module(self.model, name, new_layer)
            except Exception as e:
                logger.warning("Failed to compress %s with SVDLinear: %s", name, e)
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False

        # Post-compression forward sanity check
        forward_check_success = False
        try:
            self.model.eval()
            with torch.no_grad():
                d = next(self.model.parameters()).device
                dummy = torch.randint(0, 100, (1, 16), device=d)
                self.model(dummy)
            forward_check_success = True
        except Exception as e:
            logger.debug("SVDLinear post-compression forward sanity check exception: %s", e)

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers)
            / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats = self.get_compression_stats()
        stats.update(
            {
                "layer_errors": self.layer_errors,
                "compressed_layers": self.compressed_layers,
                "fallback_layers": self.fallback_layers,
                "skipped_layers": self.skipped_layers,
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "num_skipped": len(self.skipped_layers),
                "avg_reconstruction_error": avg_err,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "target_attention_only": self.target_attention_only,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "forward_check_success": forward_check_success,
                "method": "svd_linear_factorization",
            }
        )
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("svd_linear_factory")
@COMPRESSOR_REGISTRY.register("svd_linear_compressor_factory")
def svd_linear_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """SVD Linear compressor factory."""
    return_tuple = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    rank = kwargs.pop("rank", _cfg_get(comp_config, "svd_rank", _cfg_get(comp_config, "rank", 192)))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "max_error", 0.50))
    fallback = kwargs.pop(
        "fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)
    )
    lora_r = kwargs.pop(
        "lora_r", _cfg_get(comp_config, "lora_r", _cfg_get(comp_config, "lora_rank", 0))
    )
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    skip_attention_qkv = kwargs.pop(
        "skip_attention_qkv", _cfg_get(comp_config, "skip_attention_qkv", False)
    )
    target_attention_only = kwargs.pop(
        "target_attention_only", _cfg_get(comp_config, "target_attention_only", False)
    )
    protect_critical = kwargs.pop(
        "protect_critical", _cfg_get(comp_config, "protect_critical", True)
    )

    compressor = SVDLinearCompressor(
        model,
        rank=rank,
        max_error=max_error,
        fallback_to_original=fallback,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        skip_attention_qkv=skip_attention_qkv,
        target_attention_only=target_attention_only,
        protect_critical=protect_critical,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_tuple or kwargs.pop("return_both", False):
        return compressed_model, stats
    return compressed_model


__all__ = [
    "SVDLinearLayer",
    "SVDLinearCompressor",
    "svd_linear_factory",
    "FusedSVDLinear",
    "fused_svd_linear",
]
