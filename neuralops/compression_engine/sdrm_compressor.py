"""
SDRM: Scaled Dimension Rank Modulation.
Assigns layer rank proportional to parameter budget for edge inference and factorizes
into low-rank SVDLinear layers to achieve physical parameter reduction.
Registered as 'sdrm', 'sdrm_factory', and 'sdrm_compressor_factory'.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from neuralops.compression_engine.aa_svd_compressor import SVDLinear
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


@COMPRESSOR_REGISTRY.register("sdrm_class")
@COMPRESSOR_REGISTRY.register("sdrm_compressor_class")
class SDRMCompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    SDRM: Scaled Dimension Rank Modulation.
    Assigns layer rank proportional to parameter budget for edge inference and factorizes
    target layers into low-rank SVDLinear modules with optional LoRA fine-tuning adapters.
    """

    def __init__(
        self,
        model: nn.Module,
        total_param_ratio: float = 0.7,
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
        super().__init__(model, name="sdrm", **kwargs)
        cfg_obj = cfg or kwargs.get("comp_config")

        self.total_param_ratio = max(
            0.05,
            min(
                float(
                    total_param_ratio
                    or _cfg_get(
                        cfg_obj, "total_param_ratio", _cfg_get(cfg_obj, "ratio", 0.7)
                    )
                ),
                1.0,
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
        self.protect_output_proj: bool = bool(
            _cfg_get(cfg_obj, "protect_output_proj", True)
        )

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
        num_heads = getattr(
            cfg, "num_attention_heads", getattr(module, "num_attention_heads", None)
        )
        num_kv_heads = getattr(
            cfg, "num_key_value_heads", getattr(module, "num_key_value_heads", None)
        )
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
        PROTECTED_ALWAYS = [
            "classifier",
            "score",
            "lm_head",
            "pooler",
            "embedding",
            "cls_head",
            "head",
            "logits_processor",
        ]
        ATTENTION_OUTPUT = ["out_lin", "out_proj", "o_proj", "c_proj", "output.dense"]

        if any(k in name_l for k in PROTECTED_ALWAYS):
            self.protected_layers.append(name)
            return False

        if self.protect_output_proj and any(k in name_l for k in ATTENTION_OUTPUT):
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
        if not is_linear_like or isinstance(base_mod, SVDLinear):
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
    ) -> SVDLinear | None:
        """Decompose a single linear or Conv1D layer into SVDLinear in FP32 with validation."""
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

        norm_orig = torch.norm(W, p="fro").item()
        if norm_orig < 1e-4:
            return None

        max_r = min(out_f, in_f)
        param_cap = max(1, (in_f * out_f) // (in_f + out_f) - 1)

        if self.min_rank > max_r - 1:
            logger.warning(
                "SDRM: min_rank=%d exceeds max_r=%d for %s; skipping",
                self.min_rank,
                max_r - 1,
                cls_name,
            )
            return None

        min_rank_effective = min(self.min_rank, max_r - 1, param_cap)
        r = max(min_rank_effective, min(rank, max_r - 1, param_cap))

        if r >= max_r:
            logger.debug(
                "SDRM: rank %d >= max_r %d; skipping layer for no parameter reduction",
                r,
                max_r,
            )
            return None

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        except Exception as e:
            logger.warning("SDRM SVD decomposition failure: %s", e)
            return None

        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
        V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(dtype=dtype, device=device)

        has_base_layer = hasattr(layer, "base_layer")
        has_existing_lora = hasattr(layer, "lora_A") and hasattr(layer, "lora_B")
        effective_lora_r = (
            0
            if has_base_layer
            else (
                getattr(layer, "r", self.lora_r)
                if (has_existing_lora and getattr(layer, "r", 0) > 0)
                else self.lora_r
            )
        )

        svd_layer = SVDLinear(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=base_layer.bias is not None,
            dtype=dtype,
            device=device,
            lora_r=effective_lora_r if (self.use_lora or has_existing_lora) else 0,
            lora_alpha=self.lora_alpha,
        )
        svd_layer.set_low_rank_factors(U_r, V_r)
        if base_layer.bias is not None:
            svd_layer.bias.data.copy_(base_layer.bias.data)

        # Preserve existing LoRA adapters when replacing full module
        if not has_base_layer and has_existing_lora and svd_layer.use_lora:
            try:
                src_lA = getattr(layer.lora_A, "weight", layer.lora_A)
                src_lB = getattr(layer.lora_B, "weight", layer.lora_B)
                dst_lA = getattr(svd_layer.lora_A, "weight", svd_layer.lora_A)
                dst_lB = getattr(svd_layer.lora_B, "weight", svd_layer.lora_B)
                if (
                    hasattr(src_lA, "data")
                    and hasattr(dst_lA, "data")
                    and src_lA.shape == dst_lA.shape
                ):
                    dst_lA.data.copy_(src_lA.data)
                if (
                    hasattr(src_lB, "data")
                    and hasattr(dst_lB, "data")
                    and src_lB.shape == dst_lB.shape
                ):
                    dst_lB.data.copy_(src_lB.data)
            except Exception as e_lora:
                logger.warning(
                    "Could not copy existing LoRA adapters in SDRM: %s", e_lora
                )

        # Frobenius reconstruction error check
        W_rec = torch.matmul(U_r.float(), V_r.float())
        norm_orig_clamped = torch.norm(W, p="fro").clamp_min(1e-8)
        rel_err = (torch.norm(W - W_rec, p="fro") / norm_orig_clamped).item()

        is_invalid = torch.isnan(W_rec).any() or torch.isinf(W_rec).any()
        if rel_err > self.max_error:
            logger.warning(
                "SDRM: reconstruction error %.4f exceeded threshold %.4f.",
                rel_err,
                self.max_error,
            )
            is_invalid = True

        if is_invalid and self.fallback_to_original:
            return None

        svd_layer.reconstruction_error = rel_err
        return svd_layer

    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        """Compress linear layers using budget-proportional rank assignment."""
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
            logger.warning("SDRM: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        for name, module in target_layers:
            mod_id = id(module)
            if mod_id in self.visited_modules:
                logger.debug("SDRM: Skipping tied module %s", name)
                self.skipped_layers.append(name)
                continue
            self.visited_modules.add(mod_id)

            try:
                base_m = getattr(module, "base_layer", module)
                raw_w = base_m.weight.data
                cls_name = base_m.__class__.__name__
                is_conv1d = "conv1d" in cls_name.lower() or hasattr(base_m, "nf")
                out_f = (
                    raw_w.shape[1]
                    if is_conv1d
                    else getattr(base_m, "out_features", raw_w.shape[0])
                )
                in_f = (
                    raw_w.shape[0]
                    if is_conv1d
                    else getattr(base_m, "in_features", raw_w.shape[1])
                )

                params = in_f * out_f
                allowed = params * self.total_param_ratio
                rank = max(self.min_rank, int(allowed / max(in_f + out_f, 1)))

                svd_layer = self.compress_linear(module, rank=rank)
                if svd_layer is not None:
                    err = getattr(svd_layer, "reconstruction_error", 0.0)
                    self.layer_errors[name] = err
                    self.replaced_layers.append(
                        {"name": name, "rank": svd_layer.rank, "error": err}
                    )
                    if hasattr(module, "base_layer"):
                        module.base_layer = svd_layer
                        if hasattr(module, "lora_A") and module.lora_A is not None:
                            for p in (
                                module.lora_A.parameters()
                                if hasattr(module.lora_A, "parameters")
                                else [module.lora_A]
                            ):
                                p.requires_grad = True
                        if hasattr(module, "lora_B") and module.lora_B is not None:
                            for p in (
                                module.lora_B.parameters()
                                if hasattr(module.lora_B, "parameters")
                                else [module.lora_B]
                            ):
                                p.requires_grad = True
                    else:
                        replace_module(self.model, name, svd_layer)
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
                "SDRM post-compression forward sanity check warning: %s", ex_sanity
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
                "total_param_ratio": self.total_param_ratio,
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
                "min_rank": self.min_rank,
                "skip_attention_qkv": self.skip_attention_qkv,
                "target_attention_only": self.target_attention_only,
                "use_lora": self.use_lora,
                "lora_r": self.lora_r,
                "method": "sdrm",
            }
        )
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("sdrm")
@COMPRESSOR_REGISTRY.register("sdrm_compressor")
@COMPRESSOR_REGISTRY.register("sdrm_factory")
@COMPRESSOR_REGISTRY.register("sdrm_compressor_factory")
def sdrm_compressor_factory(
    model: Any,
    comp_config: Any = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> nn.Module | tuple[nn.Module, dict[str, Any]]:
    """SDRM compressor factory returning (compressed_model, stats) when requested."""
    total_param_ratio = kwargs.pop(
        "total_param_ratio",
        kwargs.pop(
            "ratio",
            _cfg_get(
                comp_config, "total_param_ratio", _cfg_get(comp_config, "ratio", 0.7)
            ),
        ),
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

    compressor = SDRMCompressor(
        model,
        total_param_ratio=total_param_ratio,
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


sdrm_factory = sdrm_compressor_factory


@COMPRESSOR_REGISTRY.register("inject_sdrm")
@COMPRESSOR_REGISTRY.register("sdrm_inject")
def inject_sdrm(model: Any = None, **kwargs: Any) -> Any:
    """Safe injection hook for SDRM."""
    if model is None:
        return None
    try:
        return sdrm_compressor_factory(model, **kwargs)
    except Exception as e:
        logger.error("inject_sdrm failed: %s", e)
        return None


__all__ = [
    "SDRMCompressor",
    "sdrm_compressor_factory",
]
