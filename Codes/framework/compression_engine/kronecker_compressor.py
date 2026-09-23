"""
Kronecker Product Compressor: replaces nn.Linear and Conv1D layers with KroneckerLinear.
Registered as 'kronecker_class', 'kronecker_compressor_class', and (factory)
'kronecker', 'kronecker_compressor', 'kronecker_factory'.

Bug-fix map:
  KC-FIX-01  is_custom_attention = False on compressor.
  KC-FIX-02  Removed `kronecker_linear` alias (conflicts with class name).
  KC-FIX-03  try/except around _count_params.
  KC-FIX-04  Safe device resolution.
  KC-FIX-05  avg_err computed only over compressed layers.
  KC-FIX-06  Fallback parameter accounting uses orig_layer_params (no double-count).
  KC-FIX-07  layer_ranks[name] = None for fallback; only set for accepted layers.
  KC-FIX-08  Robust forward sanity check.
  KC-FIX-09  Idempotence flag on the model.
  KC-FIX-10  skip_should_compress_filter option.
  KC-FIX-11  min_layer_dim option.
  KC-FIX-12  Debug logging for no-target diagnostic.
  KC-FIX-13  Empty target_layers explicitly means "no targets".
  KC-FIX-14  should_compress_linear wrapped in try/except.
  KC-FIX-15  layer_modes[name] only set when layer is accepted.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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


def _get_device(model: nn.Module) -> torch.device:
    """KC-FIX-04: safe device resolution."""
    p = next(iter(model.parameters()), None)
    return p.device if p is not None else torch.device("cpu")


_ATTN_NAMES = {
    "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
    "c_attn", "c_proj", "query", "key", "value",
    "q_lin", "k_lin", "v_lin", "out_lin",
}


# ===========================================================================
# Compressor
# ===========================================================================

@COMPRESSOR_REGISTRY.register("kronecker_class")
@COMPRESSOR_REGISTRY.register("kronecker_compressor_class")
class KroneckerCompressor(BaseCompressor):
    # KC-FIX-01
    is_custom_attention: bool = False

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
        skip_should_compress_filter: bool = False,
        min_layer_dim: int = 8,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model if model is not None else nn.Identity(),
            name="kronecker", **kwargs,
        )
        conf = cfg or kwargs.get("comp_config", None)

        rank = _cfg_get(conf, "kronecker_rank", _cfg_get(conf, "rank", rank))
        max_error = _cfg_get(conf, "kronecker_max_error",
                             _cfg_get(conf, "max_error", max_error))
        fallback_to_original = _cfg_get(
            conf, "kronecker_fallback_to_original",
            _cfg_get(conf, "fallback_to_original", fallback_to_original),
        )
        lora_r = _cfg_get(conf, "lora_r", _cfg_get(conf, "lora_rank", lora_r))
        lora_alpha = _cfg_get(conf, "lora_alpha", lora_alpha)
        target_attention_only = _cfg_get(
            conf, "kronecker_target_attention_only",
            _cfg_get(conf, "target_attention_only", target_attention_only),
        )
        skip_attention_qkv = _cfg_get(conf, "skip_attention_qkv", skip_attention_qkv)
        skip_should_compress_filter = _cfg_get(
            conf, "skip_should_compress_filter", skip_should_compress_filter,
        )
        min_layer_dim = _cfg_get(conf, "min_layer_dim", min_layer_dim)
        target_layers = _cfg_get(conf, "target_layers", target_layers)

        self.rank = int(rank) if rank is not None and rank > 0 else None
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.target_attention_only = bool(target_attention_only)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.skip_should_compress_filter = bool(skip_should_compress_filter)
        self.min_layer_dim = max(1, int(min_layer_dim))
        self.target_layers = target_layers
        self.cfg = conf

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, Optional[int]] = {}
        self.layer_modes: Dict[str, str] = {}
        self.stats: Dict[str, Any] = {}
        self._compressed_params: int = 0

    # ------------------------------------------------------------------
    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads:
            if any(p in name.lower() for p in
                   ("q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn")):
                return True
        return False

    # KC-FIX-12: debug-enriched target check.
    def _is_target_module(self, name: str, module: nn.Module, _debug: bool = False) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ("lora_A", "lora_B", "base_layer")):
            if _debug:
                logger.info(f"  [skip] {name}: lora/base_layer part")
            return False
        if any(skip in name_lower for skip in
               ("classifier", "score", "lm_head", "pooler",
                "embedding", "router", "gate")):
            if _debug:
                logger.info(f"  [skip] {name}: skip keyword")
            return False

        if self.target_attention_only:
            if not any(k in name_lower for k in _ATTN_NAMES):
                if _debug:
                    logger.info(f"  [skip] {name}: not attention")
                return False

        if self._is_gqa_asymmetric(name, module):
            if _debug:
                logger.info(f"  [skip] {name}: GQA asymmetric")
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_lower = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_lower or hasattr(target_mod, "nf")
        if not (is_linear or is_conv1d):
            if _debug:
                logger.info(f"  [skip] {name}: not linear-like")
            return False
        if isinstance(target_mod, KroneckerLinear):
            if _debug:
                logger.info(f"  [skip] {name}: already KroneckerLinear")
            return False

        w = getattr(target_mod, "weight", None)
        if w is None:
            if _debug:
                logger.info(f"  [skip] {name}: weight None")
            return False

        # KC-FIX-11: filter tiny layers.
        if min(w.shape[0], w.shape[1]) < self.min_layer_dim:
            if _debug:
                logger.info(f"  [skip] {name}: dim too small {tuple(w.shape)}")
            return False

        # KC-FIX-14: guard against framework errors.
        if not self.skip_should_compress_filter:
            try:
                if not should_compress_linear(self.model, name):
                    if _debug:
                        logger.info(f"  [skip] {name}: should_compress_linear=False")
                    return False
            except Exception as e:
                if _debug:
                    logger.info(f"  [skip] {name}: should_compress_linear raised {e}")
                return False

        # KC-FIX-13: empty target_layers means "no targets".
        if self.target_layers is not None:
            ok = any(t in name for t in self.target_layers)
            if not ok and _debug:
                logger.info(f"  [skip] {name}: not in target_layers")
            return ok

        if _debug:
            logger.info(f"  [TARGET] {name}: shape={tuple(w.shape)}")
        return True

    # ------------------------------------------------------------------
    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        # KC-FIX-09: idempotence.
        if getattr(self.model, "_kronecker_compressed", False):
            logger.warning("KroneckerCompressor: model already compressed; skipping.")
            try:
                return self.model, self.get_compression_stats()
            except Exception:
                return self.model, {}

        if model is not None:
            self.model = model
            try:
                self._original_params = self._count_params(self.model)
            except Exception:
                self._original_params = sum(p.numel() for p in self.model.parameters())

        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        self.layer_modes = {}
        seen_weights: Set[int] = set()

        total_orig_params = 0
        total_comp_params = 0

        # Freeze non-targets when attention_only.
        if self.target_attention_only:
            for name, module in self.model.named_modules():
                target_mod = getattr(module, "base_layer", module)
                cls_lower = target_mod.__class__.__name__.lower()
                if (isinstance(target_mod, nn.Linear)
                        or "conv1d" in cls_lower
                        or hasattr(target_mod, "nf")):
                    if not self._is_target_module(name, module):
                        for p in module.parameters():
                            p.requires_grad = False
                        self.skipped_layers.append(name)

        targets = [
            (name, mod)
            for name, mod in list(self.model.named_modules())
            if self._is_target_module(name, mod)
        ]

        if not targets:
            # KC-FIX-12: diagnostic breakdown.
            logger.warning("KroneckerCompressor: no targets found. Diagnostic...")
            n_scanned = 0
            for name, module in self.model.named_modules():
                n_scanned += 1
                self._is_target_module(name, module, _debug=True)
            logger.warning("KroneckerCompressor diagnostic: %d modules scanned, 0 targets.", n_scanned)
            try:
                return self.model, self.get_compression_stats()
            except Exception:
                return self.model, {}

        logger.info("KroneckerCompressor: %d target layers identified.", len(targets))

        for name, module in targets:
            target_mod = getattr(module, "base_layer", module)
            if not hasattr(target_mod, "weight") or target_mod.weight is None:
                continue

            w_id = id(target_mod.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            cls_lower = target_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_lower or hasattr(target_mod, "nf")
            raw_w = target_mod.weight.data

            orig_layer_params = raw_w.numel()
            bias_mod = getattr(target_mod, "bias", None)
            if bias_mod is not None:
                orig_layer_params += bias_mod.numel()
            total_orig_params += orig_layer_params

            W_target = raw_w.t().float() if is_conv1d else raw_w.float()
            W_norm = torch.norm(W_target, p="fro").item()

            new_layer: Optional[KroneckerLinear] = None
            err = 1.0
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

            # KC-FIX-15: only set mode/rank after decision.
            if new_layer is None or err > self.max_error:
                logger.warning(
                    "Kronecker: error %.4f on %s > %.4f. %s",
                    err, name, self.max_error,
                    "Falling back." if self.fallback_to_original
                    else "Keeping original weight.",
                )
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    self.layer_ranks[name] = None
                    self.layer_modes[name] = "fallback"
                    for p in module.parameters():
                        p.requires_grad = False
                    # KC-FIX-06: use orig_layer_params (no double-count).
                    total_comp_params += orig_layer_params
                    continue
                else:
                    self.fallback_layers.append(name)
                    self.layer_ranks[name] = None
                    self.layer_modes[name] = "fallback"
                    total_comp_params += orig_layer_params
                    continue

            # Accepted.
            mode = "kronecker" if getattr(new_layer, "can_decompose", False) else "svd"
            self.layer_modes[name] = mode
            self.layer_ranks[name] = getattr(new_layer, "rank", None)

            comp_layer_params = sum(p.numel() for p in new_layer.parameters())
            total_comp_params += comp_layer_params

            try:
                replace_module(self.model, name, new_layer)
                self.compressed_layers.append(name)
            except Exception as e:
                logger.warning("Kronecker: replace_module failed for %s: %s", name, e)
                self.fallback_layers.append(name)

        # KC-FIX-08: robust forward check.
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = _get_device(self.model)
                dummy = torch.randint(0, 100, (1, 8), device=dev, dtype=torch.long)
                try:
                    self.model(input_ids=dummy, attention_mask=torch.ones_like(dummy))
                except Exception:
                    try:
                        self.model(dummy)
                    except Exception as e2:
                        logger.warning("Kronecker forward check failed: %s", e2)
                        forward_check_success = False
        except Exception as e:
            logger.warning("Kronecker forward check raised: %s", e)
            forward_check_success = False

        # KC-FIX-03: safe param count.
        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        # KC-FIX-05: avg_err only over compressed layers.
        compressed_errs = [
            self.layer_errors[n] for n in self.compressed_layers
            if n in self.layer_errors
        ]
        avg_err = (
            sum(compressed_errs) / len(compressed_errs)
            if compressed_errs else 0.0
        )
        param_ratio = (
            total_comp_params / max(total_orig_params, 1)
            if total_orig_params > 0 else 1.0
        )

        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        stats.update({
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
            "fallback_layer_count": len(self.fallback_layers),
            "effective_parameter_ratio": round(param_ratio, 4),
            "method": "kronecker",
        })
        self.stats = stats

        # KC-FIX-09: write stats onto model.
        try:
            if not hasattr(self.model, "compression_stats") or self.model.compression_stats is None:
                self.model.compression_stats = {}
            self.model.compression_stats.update({
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "compression_method": "kronecker",
                "effective_parameter_ratio": param_ratio,
                "avg_reconstruction_error": avg_err,
            })
        except Exception:
            pass

        self.model._kronecker_compressed = True
        logger.info(
            "Kronecker: %d compressed, %d fallback. param_ratio=%.4f, avg_err=%.4f.",
            len(self.compressed_layers), len(self.fallback_layers),
            param_ratio, avg_err,
        )
        return self.model, stats


# ===========================================================================
# Factory (KC-FIX-02)
# ===========================================================================

@COMPRESSOR_REGISTRY.register("kronecker")
@COMPRESSOR_REGISTRY.register("kronecker_compressor")
@COMPRESSOR_REGISTRY.register("kronecker_factory")
def kronecker_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Kronecker compressor factory."""
    rank = rank or _cfg_get(comp_config, "kronecker_rank",
                            _cfg_get(comp_config, "rank", 128))
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(comp_config, "kronecker_max_error",
                 _cfg_get(comp_config, "max_error", 0.80)),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "kronecker_fallback_to_original",
                 _cfg_get(comp_config, "fallback_to_original", True)),
    )
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 16))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "kronecker_target_attention_only",
                 _cfg_get(comp_config, "target_attention_only", False)),
    )
    skip_filter = kwargs.pop(
        "skip_should_compress_filter",
        _cfg_get(comp_config, "skip_should_compress_filter", False),
    )
    min_layer_dim = kwargs.pop(
        "min_layer_dim",
        _cfg_get(comp_config, "min_layer_dim", 8),
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
        skip_should_compress_filter=skip_filter,
        min_layer_dim=min_layer_dim,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "KroneckerCompressor",
    "kronecker_factory",
]