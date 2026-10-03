"""
Direct SVD Compressor: Truncated SVD low-rank factorized layer replacement.
Registered as 'direct_svd'. Production hardened for multi-architecture compatibility
(DistilBERT, RoBERTa, GPT-2, Qwen3.5) with error validation, Conv1D support, rank bounds, and safe fallback.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear, get_dynamic_compression_rank

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract config value from None, dict, or namespace/object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj", "query", "key", "value"}


@COMPRESSOR_REGISTRY.register("direct_svd")
@COMPRESSOR_REGISTRY.register("direct_svd_compressor")
class DirectSVDCompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    Direct SVD Compressor.

    Performs direct truncated SVD on target linear and Conv1D layers:
        W ≈ U_r * Sigma_r * V_r^T
    Allocates singular energy symmetrically:
        U_factor = U_r * sqrt(Sigma_r)
        V_factor = sqrt(Sigma_r) * V_r^T
    and replaces dense layers with factorized SVDLinear modules.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error.
    - Automatic rank escalation retries when error threshold is exceeded.
    - Safe fallback to original layer if error remains high or decomposition fails.
    - Fallback and non-target layers frozen (requires_grad=False) during recovery.
    - Robust LoRALinear and Conv1D support (with transpose and adapter preservation).
    - target_attention_only parameter: restricts SVD factorization to attention projections.
    - Strict rank safeguard: enforces r < min(in_f, out_f) and parameter reduction r * (in + out) < in * out.
    - Unified critical-layer protection via should_compress_linear.
    - Full FP32 computation for numerical stability.
    - Real metrics recording: original_layer_count, compressed_layer_count, fallback_layer_count, effective_parameter_ratio.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int | None = 32,
        min_rank: int = 16,
        rank_ratio: float | None = None,
        max_error: float = 0.35,
        fallback_to_original: bool = True,
        protect_critical: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = True,
        target_attention_only: bool = False,
        max_retries: int = 2,
        target_modules: list[str] | None = None,
        cfg: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="direct_svd", **kwargs)
        cfg_obj = cfg if cfg is not None else kwargs.get("comp_config")
        self.cfg = cfg_obj

        rank = _cfg_get(cfg_obj, "direct_svd_rank", _cfg_get(cfg_obj, "svd_rank", _cfg_get(cfg_obj, "rank", rank)))
        min_rank = _cfg_get(cfg_obj, "min_rank", min_rank)
        rank_ratio = _cfg_get(cfg_obj, "direct_svd_rank_ratio", _cfg_get(cfg_obj, "rank_ratio", rank_ratio))
        max_error = _cfg_get(cfg_obj, "direct_svd_max_error", _cfg_get(cfg_obj, "max_error", max_error))
        fallback_to_original = _cfg_get(
            cfg_obj, "direct_svd_fallback_to_original", _cfg_get(cfg_obj, "fallback_to_original", fallback_to_original)
        )
        protect_critical = _cfg_get(cfg_obj, "protect_critical", protect_critical)
        lora_r = _cfg_get(cfg_obj, "lora_r", _cfg_get(cfg_obj, "lora_rank", lora_r))
        lora_alpha = _cfg_get(cfg_obj, "lora_alpha", lora_alpha)
        skip_attention_qkv = _cfg_get(cfg_obj, "skip_attention_qkv", False)
        target_attention_only = _cfg_get(
            cfg_obj, "direct_svd_target_attention_only", _cfg_get(cfg_obj, "target_attention_only", target_attention_only)
        )

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]

        self.rank = rank
        self.min_rank = max(1, int(min_rank))
        self.rank_ratio = float(rank_ratio) if rank_ratio is not None else None
        if max_error is None:
            max_error = 0.05

        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.protect_critical = bool(protect_critical)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.target_attention_only = bool(target_attention_only)
        self.max_retries = int(max_retries)
        self.target_modules = target_modules

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.layer_ranks: dict[str, int] = {}
        self.layer_errors: dict[str, float] = {}

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

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
            return False
        if self.protect_critical and any(k in name_lower for k in ["classifier", "score", "lm_head", "pooler", "embed", "logits_processor"]):
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        if not (is_linear or is_conv1d):
            return False
        if isinstance(target_mod, SVDLinear) or isinstance(module, SVDLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self._is_gqa_asymmetric(name, target_mod):
            return False

        if self.target_attention_only:
            last_part = name.split(".")[-1]
            is_attn = (last_part in _ATTN_NAMES) or any(k in name_lower for k in ("attn", "attention"))
            if not is_attn:
                return False

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def _determine_rank(self, in_f: int, out_f: int, target_r: int | None) -> int:
        """Compute target rank guaranteeing parameter reduction and strict bounds."""
        max_reduction_r = max(1, (in_f * out_f) // (in_f + out_f) - 1)
        max_r = max(1, min(in_f - 1, out_f - 1, max_reduction_r))

        ref_dim = 768.0
        if hasattr(self, "model") and hasattr(self.model, "config") and self.model.config is not None:
            cfg_obj = self.model.config
            ref_dim = float(getattr(cfg_obj, "hidden_size", getattr(cfg_obj, "d_model", getattr(cfg_obj, "n_embd", 768.0))))
        ref_dim = max(1.0, ref_dim)

        if self.rank_ratio is not None:
            r = int(round(min(in_f, out_f) * max(0.01, min(0.99, self.rank_ratio))))
        elif target_r is not None:
            r = int(round(target_r * (min(in_f, out_f) / ref_dim)))
        else:
            r = int(round(32 * (min(in_f, out_f) / ref_dim)))

        # Dynamic architecture-aware rank selection for large models
        r = get_dynamic_compression_rank(in_f, out_f, model_or_family=self.model, default_rank=r, comp_config=getattr(self, "cfg", None))
        r = max(self.min_rank, r)
        return max(1, min(r, max_r))

    def compress_linear(
        self,
        layer: nn.Module,
        rank: int | None = None,
    ) -> tuple[SVDLinear | None, float, int]:
        """
        Decompose a single linear or Conv1D layer via SVD with symmetric factor distribution.
        Returns: (svd_layer, relative_reconstruction_error, chosen_rank)
        """
        target_mod = getattr(layer, "base_layer", layer)
        cls_name = target_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        raw_w = target_mod.weight.data

        if is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.t().contiguous().float()  # [out_features, in_features]
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.float()      # [out_features, in_features]

        max_reduction_r = max(1, (in_f * out_f) // (in_f + out_f) - 1)
        max_r = max(1, min(in_f - 1, out_f - 1, max_reduction_r))

        current_r = self._determine_rank(in_f, out_f, rank)
        device = raw_w.device
        dtype = raw_w.dtype
        norm_w = torch.norm(W, p="fro").item()

        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        except Exception as e:
            logger.warning("DirectSVD SVD computation failed: %s", e)
            return None, 1.0, 0

        best_svd_layer: SVDLinear | None = None
        best_err = 1.0
        chosen_r = current_r

        # Retry with higher rank if error exceeds threshold
        for attempt in range(self.max_retries + 1):
            r = max(1, min(current_r, max_r))
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
            V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(dtype=dtype, device=device)

            W_recon = torch.matmul(U_r.float(), V_r.float())
            if norm_w < 1e-12:
                err = 0.0
            else:
                err = float(torch.norm(W - W_recon, p="fro").item() / norm_w)

            best_err = err
            chosen_r = r

            # Determine LoRA parameters to preserve or initialize
            eff_lora_r = self.lora_r
            eff_lora_alpha = self.lora_alpha
            if hasattr(layer, "lora_A") and hasattr(layer, "r"):
                eff_lora_r = getattr(layer, "r", self.lora_r)
                eff_lora_alpha = getattr(layer, "lora_alpha", self.lora_alpha)

            svd_layer = SVDLinear(
                in_features=in_f,
                out_features=out_f,
                rank=r,
                bias=target_mod.bias is not None,
                dtype=dtype,
                device=device,
                lora_r=eff_lora_r,
                lora_alpha=eff_lora_alpha,
            )
            svd_layer.set_low_rank_factors(U_r, V_r)
            if target_mod.bias is not None and svd_layer.bias is not None:
                svd_layer.bias.data.copy_(target_mod.bias.data)

            # If the original layer was a LoRALinear wrapper, copy over existing adapter weights
            if hasattr(layer, "lora_A") and hasattr(layer, "lora_B") and svd_layer.lora_A is not None:
                try:
                    if hasattr(layer.lora_A, "weight") and svd_layer.lora_A.weight.shape == layer.lora_A.weight.shape:
                        svd_layer.lora_A.weight.data.copy_(layer.lora_A.weight.data)
                    if hasattr(layer.lora_B, "weight") and svd_layer.lora_B.weight.shape == layer.lora_B.weight.shape:
                        svd_layer.lora_B.weight.data.copy_(layer.lora_B.weight.data)
                except Exception as e:
                    logger.debug("DirectSVD adapter preservation notice: %s", e)

            best_svd_layer = svd_layer

            if err <= self.max_error or r >= max_r:
                break

            if attempt < self.max_retries:
                next_r = min(int(r * 1.5), max_r)
                if next_r <= r:
                    break
                current_r = next_r

        return best_svd_layer, best_err, chosen_r

    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        """Compress target linear and Conv1D layers across the model."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_ranks = {}
        self.layer_errors = {}
        seen_weights: set[int] = set()

        total_orig_decomp_params = 0
        total_comp_decomp_params = 0

        target_layers = [
            (name, mod)
            for name, mod in list(self.model.named_modules())
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("DirectSVD: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        # Handle freezing of non-target layers when target_attention_only=True
        if self.target_attention_only:
            for name, module in list(self.model.named_modules()):
                target_mod = getattr(module, "base_layer", module)
                cls_name = target_mod.__class__.__name__.lower()
                is_linear = isinstance(target_mod, nn.Linear)
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                if is_linear or is_conv1d:
                    if not self._is_target_module(name, module):
                        for p in module.parameters():
                            p.requires_grad = False
                        self.skipped_layers.append(name)

        for name, module in target_layers:
            target_mod = getattr(module, "base_layer", module)
            if not hasattr(target_mod, "weight") or target_mod.weight is None:
                continue
            w_id = id(target_mod.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            raw_w = target_mod.weight.data
            cls_name = target_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
            if is_conv1d:
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            else:
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]

            orig_params = out_f * in_f
            total_orig_decomp_params += orig_params

            svd_layer, err, r = self.compress_linear(module, rank=self.rank)
            self.layer_errors[name] = err
            self.layer_ranks[name] = r

            if svd_layer is None or err > self.max_error:
                logger.warning(
                    "DirectSVD reconstruction error %.4f on %s exceeded threshold %.4f.",
                    err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping DirectSVD compression for %s (fallback to original and frozen).", name)
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    total_comp_decomp_params += orig_params
                    continue

            replace_module(self.model, name, svd_layer)
            self.compressed_layers.append(name)
            total_comp_decomp_params += (r * (in_f + out_f))

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
            logger.debug("DirectSVD post-compression forward sanity check exception: %s", e)

        effective_ratio = (
            total_comp_decomp_params / total_orig_decomp_params if total_orig_decomp_params > 0 else 1.0
        )
        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats.update({
            "compressed_layers": self.compressed_layers,
            "compressed_layer_count": len(self.compressed_layers),
            "fallback_layers": self.fallback_layers,
            "fallback_layer_count": len(self.fallback_layers),
            "skipped_layers": self.skipped_layers,
            "skipped_layer_count": len(self.skipped_layers),
            "original_layer_count": len(target_layers),
            "layer_ranks": self.layer_ranks,
            "layer_errors": self.layer_errors,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "target_attention_only": self.target_attention_only,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "avg_reconstruction_error": avg_err,
            "effective_parameter_ratio": round(effective_ratio, 4),
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            "forward_check_success": forward_check_success,
            "method": "direct_truncated_svd",
        })
        logger.info(
            "Compressor direct_svd finished: %d compressed, %d fallback, avg_err=%.4f, max_err_threshold=%.4f",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            self.max_error,
        )
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("direct_svd_factory")
@COMPRESSOR_REGISTRY.register("direct_svd_compressor_factory")
def direct_svd_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Direct SVD factory function."""
    return_tuple = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    rank = kwargs.pop("rank", _cfg_get(comp_config, "direct_svd_rank", _cfg_get(comp_config, "rank", 32)))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "direct_svd_max_error", _cfg_get(comp_config, "max_error", 0.35)))
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "direct_svd_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)),
    )
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", _cfg_get(comp_config, "lora_rank", 0)))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    skip_attention_qkv = kwargs.pop("skip_attention_qkv", _cfg_get(comp_config, "skip_attention_qkv", True))
    protect_critical = kwargs.pop("protect_critical", _cfg_get(comp_config, "protect_critical", True))
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "direct_svd_target_attention_only", _cfg_get(comp_config, "target_attention_only", False)),
    )

    compressor = DirectSVDCompressor(
        model,
        rank=rank,
        max_error=max_error,
        fallback_to_original=fallback,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        skip_attention_qkv=skip_attention_qkv,
        protect_critical=protect_critical,
        target_attention_only=target_attention_only,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_tuple:
        return compressed_model, stats
    return compressed_model


direct_svd_compressor_factory = direct_svd_factory

__all__ = ["DirectSVDCompressor", "direct_svd_factory", "direct_svd_compressor_factory"]

