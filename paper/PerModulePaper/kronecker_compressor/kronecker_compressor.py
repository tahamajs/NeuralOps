"""
Kronecker Product Compressor: replaces nn.Linear and Conv1D layers with KroneckerLinear.
Registered in COMPRESSOR_REGISTRY as 'kronecker'. Production hardened with
reconstruction validation, Conv1D support, adaptive SVD fallback, and critical layer protection.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

from .kronecker_linear import KroneckerLinear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
    "c_attn", "c_proj", "query", "key", "value", "q_lin", "k_lin", "v_lin", "out_lin",
}


@COMPRESSOR_REGISTRY.register("kronecker_linear_class")
@COMPRESSOR_REGISTRY.register("kronecker_compressor_class")
class KroneckerCompressor(BaseCompressor):
    """
    Compresses a model by replacing eligible nn.Linear and Conv1D layers with a
    Kronecker product decomposition (A (x) B), falling back to adaptive low-rank SVD
    when dimensions do not allow composite factorization or error exceeds max_error.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: Optional[int] = None,
        max_error: float = 0.80,
        fallback_to_original: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        target_attention_only: bool = False,
        skip_attention_qkv: bool = False,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="kronecker", **kwargs)
        conf = cfg or kwargs.get("comp_config", None)

        rank = _cfg_get(conf, "kronecker_rank", _cfg_get(conf, "rank", rank))
        max_error = _cfg_get(conf, "kronecker_max_error", _cfg_get(conf, "max_error", max_error))
        fallback_to_original = _cfg_get(
            conf, "kronecker_fallback_to_original", _cfg_get(conf, "fallback_to_original", fallback_to_original)
        )
        lora_r = _cfg_get(conf, "lora_r", _cfg_get(conf, "lora_rank", lora_r))
        lora_alpha = _cfg_get(conf, "lora_alpha", lora_alpha)
        target_attention_only = _cfg_get(
            conf, "kronecker_target_attention_only", _cfg_get(conf, "target_attention_only", target_attention_only)
        )
        skip_attention_qkv = _cfg_get(conf, "skip_attention_qkv", skip_attention_qkv)
        target_layers = _cfg_get(conf, "target_layers", target_layers)

        self.rank = int(rank) if rank is not None and rank > 0 else None
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.target_attention_only = bool(target_attention_only)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.target_layers = target_layers
        self.cfg = conf

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, Optional[int]] = {}
        self.layer_modes: Dict[str, str] = {}

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
        if any(skip in name_lower for skip in ["classifier", "score", "lm_head", "pooler", "embedding", "router", "gate"]):
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
        if not (is_linear or is_conv1d):
            return False
        if isinstance(target_mod, KroneckerLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_layers is not None:
            return any(t in name for t in self.target_layers)
        return True

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Compress the model by substituting Linear/Conv1D layers with KroneckerLinear.
        """
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        self.layer_modes = {}
        seen_weights: set[int] = set()

        total_orig_params = 0
        total_comp_params = 0

        # Freeze non-target linear/conv1d layers when target_attention_only is True
        if self.target_attention_only:
            for name, module in self.model.named_modules():
                target_mod = getattr(module, "base_layer", module)
                cls_name = target_mod.__class__.__name__.lower()
                is_linear = isinstance(target_mod, nn.Linear)
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                if (is_linear or is_conv1d) and not self._is_target_module(name, module):
                    for p in module.parameters():
                        p.requires_grad = False
                    self.skipped_layers.append(name)

        targets = [
            (name, mod)
            for name, mod in list(self.model.named_modules())
            if self._is_target_module(name, mod)
        ]

        if not targets:
            logger.warning("KroneckerCompressor: No compressible layers found.")
            return self.model, self.get_compression_stats()

        for name, module in targets:
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

            orig_layer_params = raw_w.numel()
            if getattr(target_mod, "bias", None) is not None:
                orig_layer_params += target_mod.bias.numel()
            total_orig_params += orig_layer_params

            if is_conv1d:
                W_target = raw_w.t().float()
            else:
                W_target = raw_w.float()

            W_norm = torch.norm(W_target, p="fro").item()

            try:
                new_layer = KroneckerLinear.from_linear(
                    module,
                    rank=self.rank,
                    max_error=self.max_error,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )
                W_comp = new_layer.get_weight().float()
                err = float(torch.norm(W_target - W_comp, p="fro").item() / max(W_norm, 1e-12))
            except Exception as e:
                logger.warning("KroneckerLinear factorization failed on %s: %s", name, e)
                new_layer = None
                err = 1.0

            self.layer_errors[name] = err
            mode = "kronecker" if (new_layer is not None and getattr(new_layer, "can_decompose", False)) else "svd"
            self.layer_modes[name] = mode
            self.layer_ranks[name] = getattr(new_layer, "rank", None) if new_layer is not None else None

            if new_layer is None or err > self.max_error:
                logger.warning(
                    "Kronecker reconstruction error %.4f on %s exceeded threshold %.4f.",
                    err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping Kronecker compression for %s (fallback to original).", name)
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    total_comp_params += sum(p.numel() for p in module.parameters())
                    continue

            # Compute compressed parameters for new layer
            comp_layer_params = sum(p.numel() for p in new_layer.parameters())
            total_comp_params += comp_layer_params

            replace_module(self.model, name, new_layer)
            self.compressed_layers.append(name)

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
                        (m for m in self.model.modules() if isinstance(m, (nn.Linear, KroneckerLinear))),
                        None,
                    )
                    in_dim = getattr(first_linear, "in_features", 8)
                    self.model(torch.randn(1, 8, in_dim, device=dev))
        except Exception as e:
            logger.warning("Kronecker post-compression forward check warning: %s", e)
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        param_ratio = round(total_comp_params / max(total_orig_params, 1), 4) if total_orig_params > 0 else 1.0

        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "fallback_layers": self.fallback_layers,
                "skipped_layers": self.skipped_layers,
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "num_skipped": len(self.skipped_layers),
                "layer_errors": self.layer_errors,
                "layer_ranks": self.layer_ranks,
                "layer_modes": self.layer_modes,
                "rank": self.rank,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "target_attention_only": self.target_attention_only,
                "forward_check_success": forward_check_success,
                "avg_reconstruction_error": avg_err,
                "compressed_layer_count": len(self.compressed_layers),
                "effective_parameter_ratio": param_ratio,
            }
        )
        logger.info(
            "Kronecker compression complete: %d compressed, %d fallback. Param ratio: %.4f, Avg error: %.4f.",
            len(self.compressed_layers),
            len(self.fallback_layers),
            param_ratio,
            avg_err,
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("kronecker")
@COMPRESSOR_REGISTRY.register("kronecker_linear")
@COMPRESSOR_REGISTRY.register("kronecker_compressor")
@COMPRESSOR_REGISTRY.register("kronecker_factory")
def kronecker_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Kronecker compressor factory function."""
    rank = rank or _cfg_get(comp_config, "kronecker_rank", _cfg_get(comp_config, "rank", 128))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "kronecker_max_error", _cfg_get(comp_config, "max_error", 0.80)))
    fallback = kwargs.pop("fallback_to_original", _cfg_get(
        comp_config, "kronecker_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)
    ))
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 16))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "kronecker_target_attention_only", _cfg_get(comp_config, "target_attention_only", False)),
    )
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    compressor = KroneckerCompressor(
        model,
        rank=rank,
        max_error=max_error,
        fallback_to_original=fallback,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_attention_only=target_attention_only,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


kronecker_compressor_factory = kronecker_factory

__all__ = ["KroneckerCompressor", "kronecker_factory", "kronecker_compressor_factory"]

