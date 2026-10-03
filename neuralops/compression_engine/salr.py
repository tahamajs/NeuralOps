"""
SALR: Structure-Aware / Sparse Low-Rank compression combining SVD with bottleneck pruning.
Registered in COMPRESSOR_REGISTRY as 'salr', 'salr_factory', and 'salr_compressor_factory'.
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
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


class SALRLayer(nn.Module):
    """
    Structure-Aware Low-Rank + Sparse Layer (SALR).

    Combines truncated SVD rank reduction with sparse singular component selection
    and optional trainable LoRA fine-tuning adapters:
        y = (x @ (V * mask).T) @ U.T + bias + (lora_B @ lora_A @ x) * scaling
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
        device: torch.device | None = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))
        self.sparsity = max(0.0, min(float(sparsity), 0.90))
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        requires_grad = not (self.lora_r > 0)
        self.V = nn.Parameter(
            torch.empty(self.rank, in_features, dtype=dtype, device=device),
            requires_grad=requires_grad,
        )
        self.U = nn.Parameter(
            torch.empty(out_features, self.rank, dtype=dtype, device=device),
            requires_grad=requires_grad,
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=dtype, device=device),
                requires_grad=requires_grad,
            )
        else:
            self.register_parameter("bias", None)

        # 1D binary mask on singular bottleneck components: [rank, 1]
        mask = torch.ones(self.rank, 1, dtype=torch.float32, device=device)
        if 0.0 < self.sparsity < 1.0:
            num_zeros = int(round(self.rank * self.sparsity))
            if num_zeros > 0:
                mask[-num_zeros:, :] = 0.0
        self.register_buffer("mask", mask)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty(
                    (self.lora_r, in_features), dtype=torch.float32, device=device
                )
            )
            self.lora_B = nn.Parameter(
                torch.zeros(
                    (out_features, self.lora_r), dtype=torch.float32, device=device
                )
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_low_rank_factors(
        self, U_factor: torch.Tensor, V_factor: torch.Tensor
    ) -> None:
        with torch.no_grad():
            self.U.copy_(U_factor)
            self.V.copy_(V_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Two-stage forward pass with sparse bottleneck mask and optional LoRA branch."""
        V_masked = self.V * self.mask.to(dtype=self.V.dtype)
        latent = F.linear(x, V_masked)  # [..., rank]
        out = F.linear(latent, self.U, self.bias)  # [..., out_features]

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = (
                F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scaling
            )
            out = out + lora_out.to(out.dtype)
        return out

    @property
    def weight(self) -> torch.Tensor:
        """Effective reconstructed weight matrix."""
        V_masked = self.V * self.mask.to(dtype=self.V.dtype)
        return torch.matmul(self.U, V_masked)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, sparsity={self.sparsity}, lora_r={self.lora_r}"


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@COMPRESSOR_REGISTRY.register("salr")
class SALRCompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    SALR Compressor: Structure-Aware Low-Rank and Sparse Pruning.

    Factorizes dense nn.Linear or Conv1D layers into symmetric low-rank factors
    (U * sqrt(Sigma), sqrt(Sigma) * V^T) and applies structured sparse pruning
    to trailing singular components in the bottleneck with LoRA adapters for recovery.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 64,
        sparsity: float = 0.5,
        min_rank: int = 8,
        skip_attention_qkv: bool = False,
        target_attention_only: bool = False,
        use_lora: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        target_modules: list[str] | None = None,
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        min_numel: int = 16,
        device: str | torch.device | None = None,
        cfg: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="salr", **kwargs)
        cfg_obj = cfg or kwargs.get("comp_config")

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]
        elif cfg_obj is not None:
            rank = _cfg_get(cfg_obj, "salr_rank", _cfg_get(cfg_obj, "rank", rank))

        self.rank = int(rank)
        self.sparsity = max(
            0.0,
            min(
                float(
                    sparsity
                    or _cfg_get(
                        cfg_obj, "salr_sparsity", _cfg_get(cfg_obj, "sparsity", 0.5)
                    )
                ),
                0.90,
            ),
        )
        self.min_rank = max(1, int(min_rank or _cfg_get(cfg_obj, "min_rank", 8)))
        self.skip_attention_qkv = bool(
            skip_attention_qkv
            if skip_attention_qkv is not None
            else _cfg_get(cfg_obj, "skip_attention_qkv", False)
        )
        self.target_attention_only = bool(
            target_attention_only
            if target_attention_only is not None
            else _cfg_get(cfg_obj, "target_attention_only", False)
        )
        self.use_lora = bool(
            use_lora or _cfg_get(cfg_obj, "use_lora", False) or (lora_r > 0)
        )
        self.lora_r = int(lora_r or _cfg_get(cfg_obj, "lora_r", 0))
        self.lora_alpha = float(lora_alpha or _cfg_get(cfg_obj, "lora_alpha", 1.0))
        self.target_modules = target_modules
        self.max_error = float(max_error or _cfg_get(cfg_obj, "max_error", 0.50))
        self.fallback_to_original = bool(
            fallback_to_original
            if fallback_to_original is not None
            else _cfg_get(cfg_obj, "fallback_to_original", True)
        )
        self.min_numel = max(1, int(min_numel or _cfg_get(cfg_obj, "min_numel", 16)))

        # Target device derivation
        if device is not None:
            try:
                self.target_device = torch.device(device)
            except Exception:
                self.target_device = (
                    next(model.parameters()).device
                    if list(model.parameters())
                    else torch.device("cpu")
                )
        else:
            self.target_device = (
                next(model.parameters()).device
                if list(model.parameters())
                else torch.device("cpu")
            )

        # Diagnostics & Tracking
        self.visited_modules: set[int] = set()
        self.compressed_layers: list[str] = []
        self.protected_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.failed_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.replaced_layers: list[dict[str, Any]] = []

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        """Check if layer belongs to an attention projection in a GQA architecture."""
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        is_gqa = (
            num_heads is not None
            and num_kv_heads is not None
            and num_kv_heads < num_heads
        )

        name_l = name.lower()
        is_attn_qkv = any(
            p in name_l
            for p in [
                "q_proj",
                "k_proj",
                "v_proj",
                "c_attn",
                "query",
                "key",
                "value",
                "q_lin",
                "k_lin",
                "v_lin",
                "self_attn.q",
                "self_attn.k",
                "self_attn.v",
            ]
        )
        return (is_gqa or self.skip_attention_qkv) and is_attn_qkv

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_l = name.lower()
        # 1. Protection for critical / classification / head / pooler / embedding layers
        if any(
            k in name_l
            for k in [
                "classifier",
                "score",
                "lm_head",
                "pooler",
                "embed",
                "embedding",
                "cls_head",
                "head",
                "logits_processor",
            ]
        ):
            self.protected_layers.append(name)
            return False

        # Target attention only filter
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

        # 2. GQA-aware skipping
        if self._is_gqa_asymmetric(name, module):
            self.skipped_layers.append(name)
            return False

        # 3. Check linear or Conv1D nature
        base_mod = getattr(module, "base_layer", module)
        cls_name = base_mod.__class__.__name__
        raw_w = getattr(base_mod, "weight", None)
        if raw_w is None or not isinstance(raw_w, torch.Tensor):
            return False

        is_conv1d = "conv1d" in cls_name.lower() or hasattr(base_mod, "nf")
        is_linear_like = isinstance(base_mod, nn.Linear) or is_conv1d
        if not is_linear_like or isinstance(base_mod, SALRLayer):
            return False

        # 4. Tiny layer guard
        if raw_w.numel() < self.min_numel:
            self.skipped_layers.append(name)
            return False

        # 5. Respect framework compatibility rules
        if not should_compress_linear(self.model, name):
            self.protected_layers.append(name)
            return False

        # 6. User-defined explicit target modules
        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def compress_linear(
        self,
        layer: nn.Module,
        rank: int,
    ) -> SALRLayer | None:
        """Decompose a single linear or Conv1D layer into SALRLayer with validation."""
        base_layer = getattr(layer, "base_layer", layer)
        raw_w = base_layer.weight.data
        cls_name = base_layer.__class__.__name__
        is_conv1d = "conv1d" in cls_name.lower() or hasattr(base_layer, "nf")

        if is_conv1d:
            W = raw_w.t().contiguous().float()
            out_f, in_f = W.shape
        else:
            W = raw_w.contiguous().float()
            out_f = getattr(base_layer, "out_features", W.shape[0])
            in_f = getattr(base_layer, "in_features", W.shape[1])

        max_r = min(out_f, in_f)
        param_cap = max(1, (in_f * out_f) // (in_f + out_f) - 1)
        r = max(self.min_rank, min(rank, max_r - 1, param_cap))
        if r >= max_r:
            logger.debug(
                "SALR: rank %d >= max_r %d; skipping layer for no parameter reduction",
                r,
                max_r,
            )
            return None

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # SVD in FP32
        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        except Exception as e:
            logger.warning("SALR SVD decomposition failure: %s", e)
            return None

        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
        V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(dtype=dtype, device=device)

        has_existing_lora = hasattr(layer, "lora_A") and hasattr(layer, "lora_B")
        effective_lora_r = self.lora_r
        if has_existing_lora and getattr(layer, "r", 0) > 0:
            effective_lora_r = getattr(layer, "r", self.lora_r)

        salr_layer = SALRLayer(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            sparsity=self.sparsity,
            bias=base_layer.bias is not None,
            dtype=dtype,
            device=device,
            lora_r=effective_lora_r if (self.use_lora or has_existing_lora) else 0,
            lora_alpha=self.lora_alpha,
        )
        salr_layer.set_low_rank_factors(U_r, V_r)
        if base_layer.bias is not None:
            salr_layer.bias.data.copy_(base_layer.bias.data)

        # Preserve existing LoRA adapters
        if has_existing_lora and salr_layer.lora_r > 0:
            try:
                if hasattr(layer.lora_A, "weight") and salr_layer.lora_A is not None:
                    salr_layer.lora_A.data.copy_(layer.lora_A.weight.data)
                elif (
                    hasattr(layer, "lora_A")
                    and isinstance(layer.lora_A, torch.Tensor)
                    and salr_layer.lora_A is not None
                ):
                    salr_layer.lora_A.data.copy_(layer.lora_A.data)
                if hasattr(layer.lora_B, "weight") and salr_layer.lora_B is not None:
                    salr_layer.lora_B.data.copy_(layer.lora_B.weight.data)
                elif (
                    hasattr(layer, "lora_B")
                    and isinstance(layer.lora_B, torch.Tensor)
                    and salr_layer.lora_B is not None
                ):
                    salr_layer.lora_B.data.copy_(layer.lora_B.data)
            except Exception as e_lora:
                logger.warning("Could not copy existing LoRA adapters: %s", e_lora)

        # Frobenius reconstruction error check
        W_rec = salr_layer.weight.float()
        norm_orig = torch.norm(W, p="fro").clamp_min(1e-8)
        rel_err = (torch.norm(W - W_rec, p="fro") / norm_orig).item()

        is_invalid = torch.isnan(W_rec).any() or torch.isinf(W_rec).any()
        if rel_err > self.max_error:
            logger.warning(
                "SALR: reconstruction error %.4f exceeded threshold %.4f.",
                rel_err,
                self.max_error,
            )
            is_invalid = True

        if is_invalid and self.fallback_to_original:
            return None

        salr_layer.reconstruction_error = rel_err
        return salr_layer

    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        """Compress all linear layers in the model."""
        self.layer_errors = {}
        self.replaced_layers = []
        self.visited_modules.clear()
        self.compressed_layers.clear()
        self.protected_layers.clear()
        self.skipped_layers.clear()
        self.fallback_layers.clear()
        self.failed_layers.clear()

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("SALR: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        for name, module in target_layers:
            mod_id = id(module)
            if mod_id in self.visited_modules:
                logger.debug("SALR: Skipping tied module %s", name)
                self.skipped_layers.append(name)
                continue
            self.visited_modules.add(mod_id)

            try:
                new_layer = self.compress_linear(module, rank=self.rank)
                if new_layer is not None:
                    err = getattr(new_layer, "reconstruction_error", 0.0)
                    self.layer_errors[name] = err
                    self.replaced_layers.append(
                        {"name": name, "rank": new_layer.rank, "error": err}
                    )
                    replace_module(self.model, name, new_layer)
                    self.compressed_layers.append(name)
                else:
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
            except Exception as e_comp:
                logger.warning("Failed to compress %s: %s", name, e_comp)
                self.failed_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False

        # Post-compression forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dummy_input = torch.randint(0, 100, (1, 8), device=self.target_device)
                self.model(dummy_input)
        except Exception as ex_sanity:
            logger.warning(
                "SALR post-compression forward sanity check warning: %s", ex_sanity
            )
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            (sum(self.layer_errors.values()) / len(self.layer_errors))
            if self.layer_errors
            else 0.0
        )
        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "compressed_layer_count": len(self.compressed_layers),
                "protected_layers": self.protected_layers,
                "protected_layer_count": len(self.protected_layers),
                "skipped_layers": self.skipped_layers,
                "skipped_layer_count": len(self.skipped_layers),
                "fallback_layers": self.fallback_layers,
                "fallback_layer_count": len(self.fallback_layers),
                "failed_layers": self.failed_layers,
                "failed_layer_count": len(self.failed_layers),
                "layer_errors": self.layer_errors,
                "avg_reconstruction_error": avg_err,
                "forward_check_success": forward_check_success,
                "rank": self.rank,
                "min_rank": self.min_rank,
                "sparsity": self.sparsity,
                "skip_attention_qkv": self.skip_attention_qkv,
                "target_attention_only": self.target_attention_only,
                "use_lora": self.use_lora,
                "lora_r": self.lora_r,
                "method": "salr",
            }
        )
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("salr_factory")
@COMPRESSOR_REGISTRY.register("salr_compressor_factory")
def salr_factory(
    model: nn.Module,
    comp_config: Any = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> nn.Module | tuple[nn.Module, dict[str, Any]]:
    """SALR factory function returning (compressed_model, stats) when requested."""
    rank = kwargs.pop(
        "rank", _cfg_get(comp_config, "salr_rank", _cfg_get(comp_config, "rank", 64))
    )
    sparsity = kwargs.pop(
        "sparsity",
        _cfg_get(comp_config, "salr_sparsity", _cfg_get(comp_config, "sparsity", 0.5)),
    )
    min_rank = kwargs.pop("min_rank", _cfg_get(comp_config, "min_rank", 8))
    skip_attention_qkv = kwargs.pop(
        "skip_attention_qkv", _cfg_get(comp_config, "skip_attention_qkv", False)
    )
    target_attention_only = kwargs.pop(
        "target_attention_only", _cfg_get(comp_config, "target_attention_only", False)
    )
    use_lora = kwargs.pop("use_lora", _cfg_get(comp_config, "use_lora", False))
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "max_error", 0.50))
    fallback = kwargs.pop(
        "fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)
    )

    compressor = SALRCompressor(
        model,
        rank=rank,
        sparsity=sparsity,
        min_rank=min_rank,
        skip_attention_qkv=skip_attention_qkv,
        target_attention_only=target_attention_only,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_error=max_error,
        fallback_to_original=fallback,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats or return_tuple or kwargs.pop("return_both", False):
        return compressed_model, stats
    return compressed_model


__all__ = [
    "SALRLayer",
    "SALRCompressor",
    "salr_factory",
]
