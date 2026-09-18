"""
Spectral Residual Low-Rank Adaptation (Res-SVD / Spectral Residual LoRA).

Decomposes dense weight matrices into:
1. Frozen Base Low-Rank Factors: $W_{\\text{base}} = U_r \\Sigma_r V_r^\\top$ (rank $r$, frozen).
2. Trainable Residual Adapters: Initialized directly from the top-$k$ singular directions of the
   residual error $E = W - W_{\\text{base}} = \\sum_{j=r+1}^{r+k} \\sigma_j u_j v_j^\\top$.
   $$A = \\sqrt{\\Sigma_{\\text{tail}}} V_{\\text{tail}}^\\top, \\quad B = U_{\\text{tail}} \\sqrt{\\Sigma_{\\text{tail}}}$$
   ensuring $\\Delta W = B A \\approx E$ at zero-shot initialization.
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


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Helper to safely retrieve a configuration value from dict, object, or None."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class ResSVDLinear(nn.Module):
    is_custom_attention: bool = True
    """
    Spectral Residual LoRA (Res-SVD) Linear Layer.

    Executes forward pass as:
    y = x @ W_base^T + scaling * (x @ A^T) @ B^T + bias
    where W_base is either factorized as (base_V, base_U) with requires_grad=False,
    and (lora_A, lora_B) are trainable adapter matrices initialized from the residual tail.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        base_rank: int = 32,
        residual_rank: int = 8,
        scaling: float = 1.0,
        factorize_base: bool = True,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        max_possible_rank = min(in_features, out_features)

        # Bounds validation
        self.base_rank = max(1, min(base_rank, max_possible_rank))
        remaining_rank = max(0, max_possible_rank - self.base_rank)
        self.residual_rank = max(0, min(residual_rank, remaining_rank))
        self.scaling = float(scaling)
        self.factorize_base = factorize_base

        # Base parameters (Frozen)
        if self.factorize_base:
            self.base_V = nn.Parameter(
                torch.empty(self.base_rank, in_features, dtype=dtype, device=device),
                requires_grad=False,
            )
            self.base_U = nn.Parameter(
                torch.empty(out_features, self.base_rank, dtype=dtype, device=device),
                requires_grad=False,
            )
        else:
            self.base_weight = nn.Parameter(
                torch.empty(out_features, in_features, dtype=dtype, device=device),
                requires_grad=False,
            )

        # Trainable Residual LoRA Adapters
        if self.residual_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty(self.residual_rank, in_features, dtype=dtype, device=device),
                requires_grad=True,
            )
            self.lora_B = nn.Parameter(
                torch.empty(out_features, self.residual_rank, dtype=dtype, device=device),
                requires_grad=True,
            )
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming uniform parameter initialization for base and zero for adapters."""
        if self.factorize_base:
            nn.init.kaiming_uniform_(self.base_V, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.base_U, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        if self.residual_rank > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        if self.factorize_base:
            base = torch.matmul(self.base_U, self.base_V)
        else:
            base = self.base_weight
        if self.residual_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            res = self.scaling * torch.matmul(self.lora_B, self.lora_A)
            return base + res
        return base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward execution:
        y = Base(x) + scaling * (x @ A^T) @ B^T + bias
        """
        # RESEARCH FIX: Two-stage factorized forward for frozen base to minimize memory footprint
        if self.factorize_base:
            base_intermediate = F.linear(x, self.base_V)
            base_out = F.linear(base_intermediate, self.base_U)
        else:
            base_out = F.linear(x, self.base_weight)

        # Trainable spectral residual LoRA adapter forward
        if self.residual_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            res_intermediate = F.linear(x, self.lora_A)
            res_out = self.scaling * F.linear(res_intermediate, self.lora_B)
            out = base_out + res_out
        else:
            out = base_out

        if self.bias is not None:
            out = out + self.bias

        return out

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        base_rank: int = 32,
        residual_rank: int = 8,
        scaling: float = 1.0,
        factorize_base: bool = True,
        freeze_base: bool = True,
    ) -> "ResSVDLinear":
        """
        Construct ResSVDLinear by decomposing a pretrained nn.Linear, Conv1D, or LoRALinear base_layer weight:
        1. Base factors U_r, V_r from the dominant top-r singular components.
        2. Residual adapter factors A, B directly initialized from the residual tail (r+1 to r+k).
        Preserves existing LoRA adapters if source is a LoRALinear wrapper.
        """
        base_mod = getattr(linear, "base_layer", linear)
        raw_w = base_mod.weight.data
        cls_name = base_mod.__class__.__name__
        is_conv1d = "conv1d" in cls_name.lower() or (
            raw_w.ndim == 2
            and hasattr(base_mod, "in_features")
            and base_mod.in_features == raw_w.shape[0]
            and raw_w.shape[0] < raw_w.shape[1]
        )

        if is_conv1d:
            W = raw_w.t().contiguous().float()
            out_f, in_f = W.shape
        else:
            W = raw_w.float()
            out_f, in_f = W.shape

        max_rank = min(in_f, out_f)
        param_cap = max(1, (in_f * out_f) // (in_f + out_f) - 1) if (factorize_base and in_f * out_f > 16) else max_rank

        r = max(1, min(base_rank, max_rank, param_cap))
        rem = max(0, max_rank - r)
        k = max(0, min(residual_rank, rem))

        new_layer = cls(
            in_features=in_f,
            out_features=out_f,
            base_rank=r,
            residual_rank=k,
            scaling=scaling,
            factorize_base=factorize_base,
            bias=getattr(base_mod, "bias", None) is not None,
            dtype=base_mod.weight.dtype,
            device=base_mod.weight.device,
        )

        # SVD decomposition of source base weight
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # 1. Base Truncation (Dominant Top-r Singular Components)
        U_base = U[:, :r] * torch.sqrt(S[:r]).unsqueeze(0)  # (out_f, r)
        V_base = torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]  # (r, in_f)

        if factorize_base:
            new_layer.base_U.data.copy_(U_base.to(dtype=base_mod.weight.dtype))
            new_layer.base_V.data.copy_(V_base.to(dtype=base_mod.weight.dtype))
            new_layer.base_U.requires_grad = not freeze_base
            new_layer.base_V.requires_grad = not freeze_base
        else:
            W_base = (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]
            new_layer.base_weight.data.copy_(W_base.to(dtype=base_mod.weight.dtype))
            new_layer.base_weight.requires_grad = not freeze_base

        # 2. Residual Tail Initialization or source LoRA copying
        has_src_lora = hasattr(linear, "lora_A") and hasattr(linear, "lora_B") and linear.lora_A is not None and linear.lora_B is not None
        if has_src_lora and k > 0:
            src_A = getattr(linear.lora_A, "weight", linear.lora_A)
            src_B = getattr(linear.lora_B, "weight", linear.lora_B)
            if src_A.shape == new_layer.lora_A.shape and src_B.shape == new_layer.lora_B.shape:
                new_layer.lora_A.data.copy_(src_A.data)
                new_layer.lora_B.data.copy_(src_B.data)
                new_layer.lora_A.requires_grad = True
                new_layer.lora_B.requires_grad = True
                has_src_lora = True
            else:
                has_src_lora = False

        if not has_src_lora and k > 0:
            S_tail = S[r : r + k]
            U_tail = U[:, r : r + k]
            Vh_tail = Vh[r : r + k, :]

            # Scale S_tail by adapter scaling factor gamma: (B @ A) * gamma = E_tail
            scale_factor = max(1e-8, float(scaling))
            S_tail_scaled = S_tail / scale_factor

            sqrt_S = torch.sqrt(torch.clamp(S_tail_scaled, min=0.0))
            lora_A_init = sqrt_S.unsqueeze(1) * Vh_tail  # (k, in_f)
            lora_B_init = U_tail * sqrt_S.unsqueeze(0)  # (out_f, k)

            new_layer.lora_A.data.copy_(lora_A_init.to(dtype=base_mod.weight.dtype))
            new_layer.lora_B.data.copy_(lora_B_init.to(dtype=base_mod.weight.dtype))
            new_layer.lora_A.requires_grad = True
            new_layer.lora_B.requires_grad = True

        # Copy bias if present
        if base_mod.bias is not None:
            new_layer.bias.data.copy_(base_mod.bias.data)

        return new_layer

    def merge_weights(self) -> nn.Linear:
        """
        Merge frozen base weights and trainable residual adapter weights into a dense nn.Linear.

        Returns:
            nn.Linear: Standard dense linear layer with merged weights.
        """
        device = next(self.parameters()).device
        dtype = (
            self.base_U.dtype if self.factorize_base else self.base_weight.dtype
        )

        merged_linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=device,
            dtype=dtype,
        )

        with torch.no_grad():
            if self.factorize_base:
                W_base = self.base_U.float() @ self.base_V.float()
            else:
                W_base = self.base_weight.float()

            if self.residual_rank > 0 and self.lora_A is not None and self.lora_B is not None:
                delta_W = self.scaling * (self.lora_B.float() @ self.lora_A.float())
                W_total = W_base + delta_W
            else:
                W_total = W_base

            merged_linear.weight.data.copy_(W_total.to(dtype=dtype))
            if self.bias is not None:
                merged_linear.bias.data.copy_(self.bias.data)

        return merged_linear

    def get_residual_error_norm(
        self, original_weight: Optional[torch.Tensor] = None
    ) -> float:
        """
        Calculate relative Frobenius norm reconstruction error:
        || W_orig - (W_base + scaling * B @ A) ||_F / || W_orig ||_F
        """
        with torch.no_grad():
            if self.factorize_base:
                W_approx = self.base_U.float() @ self.base_V.float()
            else:
                W_approx = self.base_weight.float()

            if self.residual_rank > 0 and self.lora_A is not None and self.lora_B is not None:
                W_approx = W_approx + self.scaling * (
                    self.lora_B.float() @ self.lora_A.float()
                )

            if original_weight is not None:
                W_orig = original_weight.float().to(W_approx.device)
                error = torch.norm(W_orig - W_approx, p="fro") / (
                    torch.norm(W_orig, p="fro") + 1e-12
                )
                return float(error.item())

            return 0.0

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"base_rank={self.base_rank}, residual_rank={self.residual_rank}, "
            f"scaling={self.scaling}, factorize_base={self.factorize_base}"
        )


class ResSVDCompressor(BaseCompressor):
    """
    Spectral Residual LoRA (Res-SVD) Compressor.

    Replaces standard Linear layers with ResSVDLinear modules, truncating base weights
    at rank r (frozen) and initializing adapter matrices A, B directly from the residual error tail.
    """

    def __init__(
        self,
        model: nn.Module,
        base_rank: int = 32,
        residual_rank: int = 8,
        scaling: float = 1.0,
        factorize_base: bool = True,
        freeze_base: bool = True,
        target_modules: Optional[List[str]] = None,
        max_error: float = 0.35,
        fallback_to_original: bool = False,
        target_attention_only: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="res_svd", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))

        base_rank = _cfg_get(cfg, "base_rank", _cfg_get(cfg, "rank", base_rank))
        residual_rank = _cfg_get(cfg, "residual_rank", residual_rank)
        scaling = _cfg_get(cfg, "scaling", _cfg_get(cfg, "lora_alpha", scaling))
        factorize_base = _cfg_get(cfg, "factorize_base", factorize_base)
        freeze_base = _cfg_get(cfg, "freeze_base", freeze_base)
        max_error = _cfg_get(cfg, "max_error", max_error)
        fallback_to_original = _cfg_get(cfg, "fallback_to_original", fallback_to_original)
        target_attention_only = bool(target_attention_only or _cfg_get(cfg, "target_attention_only", False))

        if "rank" in kwargs and kwargs["rank"] is not None:
            base_rank = kwargs["rank"]

        self.base_rank = base_rank
        self.residual_rank = residual_rank
        self.scaling = scaling
        self.factorize_base = factorize_base
        self.freeze_base = freeze_base
        self.target_modules = target_modules
        self.target_attention_only = target_attention_only
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.replaced_layers: List[Dict[str, Any]] = []

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """Check whether module is an eligible nn.Linear, Conv1D, or LoRALinear layer matching filter."""
        name_l = name.lower()
        if any(k in name_l for k in ["classifier", "score", "lm_head", "pooler", "embedding"]):
            return False

        if self.target_attention_only:
            attn_keywords = ["attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj", "query", "key", "value", "dense"]
            if not any(k in name_l for k in attn_keywords):
                return False

        base_mod = getattr(module, "base_layer", module)
        cls_name = base_mod.__class__.__name__
        is_linear_like = isinstance(base_mod, nn.Linear) or "conv1d" in cls_name.lower()
        if not is_linear_like:
            return False
        if isinstance(module, ResSVDLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Compress target linear layers into ResSVDLinear modules with Frobenius validation.
        """
        self.replaced_layers = []
        fallback_layers = []

        # Find target linear layers
        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("No compressible linear layers found in model.")
            stats = self.get_compression_stats()
            stats.update({
                "target_attention_only": self.target_attention_only,
                "forward_check_success": True,
            })
            return self.model, stats

        for name, old_linear in target_layers:
            base_mod = getattr(old_linear, "base_layer", old_linear)
            raw_w = base_mod.weight.data
            cls_name = base_mod.__class__.__name__
            is_conv1d = "conv1d" in cls_name.lower() or (
                raw_w.ndim == 2
                and hasattr(base_mod, "in_features")
                and base_mod.in_features == raw_w.shape[0]
                and raw_w.shape[0] < raw_w.shape[1]
            )
            W_orig = raw_w.t().contiguous().float() if is_conv1d else raw_w.float()

            res_layer = ResSVDLinear.from_linear(
                linear=old_linear,
                base_rank=self.base_rank,
                residual_rank=self.residual_rank,
                scaling=self.scaling,
                factorize_base=self.factorize_base,
                freeze_base=self.freeze_base,
            )

            err = res_layer.get_residual_error_norm(W_orig)
            if self.fallback_to_original and err > self.max_error:
                fallback_layers.append(name)
                for p in old_linear.parameters():
                    p.requires_grad = False
                continue

            if hasattr(old_linear, "base_layer"):
                old_linear.base_layer = res_layer
                if hasattr(old_linear, "lora_A") and old_linear.lora_A is not None:
                    for p in (old_linear.lora_A.parameters() if hasattr(old_linear.lora_A, "parameters") else [old_linear.lora_A]):
                        p.requires_grad = True
                if hasattr(old_linear, "lora_B") and old_linear.lora_B is not None:
                    for p in (old_linear.lora_B.parameters() if hasattr(old_linear.lora_B, "parameters") else [old_linear.lora_B]):
                        p.requires_grad = True
            else:
                replace_module(self.model, name, res_layer)

            self.replaced_layers.append({
                "name": name,
                "relative_error": float(err),
                "base_rank": res_layer.base_rank,
                "residual_rank": res_layer.residual_rank,
            })

        # Forward sanity check
        forward_check_success = True
        try:
            device = next(self.model.parameters()).device
            with torch.no_grad():
                dummy_input = torch.randint(0, 100, (1, 8), device=device)
                self.model(dummy_input)
        except Exception as ex_sanity:
            logger.warning("ResSVD post-compression forward sanity check warning: %s", ex_sanity)
            forward_check_success = False

        # Calculate statistics
        total_params = self._count_params(self.model)
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        frozen_params = total_params - trainable_params

        stats = self.get_compression_stats()
        stats["trainable_parameters"] = trainable_params
        stats["frozen_parameters"] = frozen_params
        stats["trainable_percentage"] = (
            (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
        )
        stats["base_rank"] = self.base_rank
        stats["residual_rank"] = self.residual_rank
        stats["replaced_layers"] = self.replaced_layers
        stats["fallback_layers"] = fallback_layers
        stats["target_attention_only"] = self.target_attention_only
        stats["forward_check_success"] = forward_check_success

        return self.model, stats


# Alias class for PEFT terminology
class SpectralResidualLoRA(ResSVDCompressor):
    """PEFT-focused wrapper for Spectral Residual LoRA."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.name = "spectral_residual_lora"


# Register under both canonical and alias names
@COMPRESSOR_REGISTRY.register("res_svd")
class ResSVDCompressorRegistered(ResSVDCompressor):
    pass


@COMPRESSOR_REGISTRY.register("spectral_residual_lora")
class SpectralResidualLoRARegistered(SpectralResidualLoRA):
    pass


def res_svd_factory(
    model: Any,
    comp_config: Any = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Res-SVD factory function."""
    compressor = ResSVDCompressor(model, comp_config=comp_config, **kwargs)
    compressed_model, stats = compressor.compress()
    if return_stats or return_tuple or kwargs.pop("return_both", False):
        return compressed_model, stats
    return compressed_model


def spectral_residual_lora_factory(
    model: Any,
    comp_config: Any = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Spectral Residual LoRA factory function."""
    compressor = SpectralResidualLoRA(model, comp_config=comp_config, **kwargs)
    compressed_model, stats = compressor.compress()
    if return_stats or return_tuple or kwargs.pop("return_both", False):
        return compressed_model, stats
    return compressed_model


res_svd_compressor_factory = res_svd_factory
SpectralResidualLoRACompressor = SpectralResidualLoRA
SVDLinear = ResSVDLinear

__all__ = [
    "ResSVDLinear",
    "SVDLinear",
    "ResSVDCompressor",
    "SpectralResidualLoRA",
    "SpectralResidualLoRACompressor",
    "res_svd_factory",
    "spectral_residual_lora_factory",
    "res_svd_compressor_factory",
]
