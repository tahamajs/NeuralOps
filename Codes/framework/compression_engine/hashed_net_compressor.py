"""
HashedNet Compressor: replaces nn.Linear and Conv1D layers with HashedLinear.
Registered in COMPRESSOR_REGISTRY as 'hashed_net', 'hashednets', and 'hashednets_compressor'.
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

from .hashed_linear import HashedLinear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract config value from None, dict, or namespace/object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj", "query", "key", "value"}


@COMPRESSOR_REGISTRY.register("hashed_net_class")
@COMPRESSOR_REGISTRY.register("hashednets_class")
class HashedNetCompressor(BaseCompressor):
    """
    Compress a model by replacing target linear/Conv1D layers with HashedLinear,
    which shares weights globally through compact hash tables.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        n_buckets: Optional[int] = None,
        compression_ratio: Optional[float] = None,
        rank: Optional[Union[int, float]] = None,
        max_error: float = 0.35,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        max_retries: int = 2,
        use_sign: bool = True,
        lora_rank: Optional[int] = None,
        lora_alpha: float = 1.0,
        freeze_hash_table: bool = False,
        skip_attention_qkv: bool = False,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="hashed_net", **kwargs)
        conf = cfg or kwargs.get("comp_config", None)
        self.cfg = conf

        n_buckets = _cfg_get(conf, "hashed_net_buckets", _cfg_get(conf, "n_buckets", n_buckets))
        compression_ratio = _cfg_get(conf, "compression_ratio", _cfg_get(conf, "target_param_ratio", compression_ratio))
        rank = _cfg_get(conf, "hashed_net_rank", _cfg_get(conf, "rank", rank))
        self.max_error = float(_cfg_get(conf, "hashed_net_max_error", _cfg_get(conf, "max_error", max_error)))
        self.target_attention_only = bool(
            _cfg_get(conf, "hashed_net_target_attention_only", _cfg_get(conf, "target_attention_only", target_attention_only))
        )
        self.fallback_to_original = bool(
            _cfg_get(conf, "hashed_net_fallback_to_original", _cfg_get(conf, "fallback_to_original", fallback_to_original))
        )
        self.use_sign = bool(_cfg_get(conf, "use_sign", use_sign))
        self.lora_rank = _cfg_get(conf, "lora_rank", _cfg_get(conf, "lora_r", _cfg_get(conf, "hashed_net_lora_rank", lora_rank)))
        if self.lora_rank is not None:
            self.lora_rank = int(self.lora_rank) if int(self.lora_rank) > 0 else None
        self.lora_alpha = float(_cfg_get(conf, "lora_alpha", _cfg_get(conf, "hashed_net_lora_alpha", lora_alpha)))
        self.freeze_hash_table = bool(_cfg_get(conf, "freeze_hash_table", freeze_hash_table or (self.lora_rank is not None and self.lora_rank > 0)))
        self.skip_attention_qkv = bool(_cfg_get(conf, "skip_attention_qkv", skip_attention_qkv))
        self.target_layers = target_layers or _cfg_get(conf, "target_layers", None)
        self.max_retries = int(max_retries)

        # Parse bucket budget or ratio
        self.compression_ratio: Optional[float] = None
        if compression_ratio is not None:
            self.compression_ratio = float(compression_ratio)
        elif rank is not None and 0.0 < float(rank) <= 1.0:
            self.compression_ratio = float(rank)

        if n_buckets is not None:
            self.base_buckets: Optional[int] = max(16, int(n_buckets))
        elif rank is not None and float(rank) > 1.0:
            self.base_buckets = max(16, int(rank) * 32)
        else:
            self.base_buckets = 65536 if self.compression_ratio is None else None

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_buckets: Dict[str, int] = {}

    def _determine_layer_buckets(self, in_features: int, out_features: int) -> int:
        total_w = in_features * out_features
        if self.compression_ratio is not None:
            target = max(16, int(total_w * self.compression_ratio))
        elif self.base_buckets is not None:
            target = min(self.base_buckets, max(16, int(total_w * 0.5)))
        else:
            target = min(65536, max(16, int(total_w * 0.25)))
        return max(16, min(target, total_w))

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
        if not (is_linear or is_conv1d) or isinstance(target_mod, HashedLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_layers is not None:
            return any(t in name for t in self.target_layers)
        return True

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress the target model by substituting linear/Conv1D layers with HashedLinear."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_buckets = {}
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

        targets = [
            (name, mod)
            for name, mod in list(self.model.named_modules())
            if self._is_target_module(name, mod)
        ]

        if not targets:
            logger.warning("HashedNet: No compressible layers found.")
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

            if hasattr(target_mod, "nx") and hasattr(target_mod, "nf"):
                in_f = int(target_mod.nx)
                out_f = int(target_mod.nf)
                W_target = raw_w.t().float()
            elif hasattr(target_mod, "in_features") and hasattr(target_mod, "out_features"):
                in_f = int(target_mod.in_features)
                out_f = int(target_mod.out_features)
                if raw_w.shape == (in_f, out_f):
                    W_target = raw_w.t().float()
                else:
                    W_target = raw_w.view(out_f, in_f).float()
            elif is_conv1d:
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
                W_target = raw_w.t().float()
            else:
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]
                W_target = raw_w.float()

            orig_layer_params = in_f * out_f
            total_orig_params += orig_layer_params
            W_norm = torch.norm(W_target, p="fro").item()
            total_weights = in_f * out_f
            max_allowed_buckets = max(16, int(total_weights * 0.90))

            init_buckets = self._determine_layer_buckets(in_f, out_f)
            current_buckets = init_buckets
            layer_compressed: Optional[HashedLinear] = None
            final_err = 1.0

            for attempt in range(self.max_retries + 1):
                try:
                    candidate = HashedLinear.from_linear(
                        module,
                        n_buckets=current_buckets,
                        use_sign=self.use_sign,
                        lora_rank=self.lora_rank,
                        lora_alpha=self.lora_alpha,
                    )
                except Exception as e:
                    logger.warning("HashedLinear instantiation failed on %s: %s", name, e)
                    break

                # Check for numerical validity
                if (
                    torch.isnan(candidate.weight_table).any()
                    or torch.isinf(candidate.weight_table).any()
                ):
                    logger.warning("HashedLinear weight table contains NaN/Inf on %s", name)
                    break

                # Measure relative Frobenius reconstruction error
                W_recon = candidate.get_weight().float()
                if W_norm < 1e-12:
                    err = 0.0
                else:
                    err = float(torch.norm(W_target - W_recon, p="fro").item() / W_norm)

                final_err = err
                layer_compressed = candidate

                if err <= self.max_error:
                    break

                # Adaptive bucket increase on error threshold exceedance
                if attempt < self.max_retries:
                    next_buckets = min(int(current_buckets * 2), max_allowed_buckets)
                    if next_buckets <= current_buckets:
                        break
                    logger.debug(
                        "HashedNet: Error %.4f > %.4f on %s. Retrying with %d buckets.",
                        err,
                        self.max_error,
                        name,
                        next_buckets,
                    )
                    current_buckets = next_buckets

            self.layer_errors[name] = final_err
            self.layer_buckets[name] = current_buckets

            if layer_compressed is None or final_err > self.max_error:
                logger.warning(
                    "HashedNet reconstruction error %.4f on %s exceeded threshold %.4f.",
                    final_err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("Skipping HashedNet compression for %s (fallback to original).", name)
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    total_comp_params += orig_layer_params
                    continue

            if self.freeze_hash_table and layer_compressed is not None:
                layer_compressed.weight_table.requires_grad = False

            try:
                if hasattr(module, "base_layer"):
                    module.base_layer = layer_compressed
                    if hasattr(module, "lora_A") and module.lora_A is not None:
                        for p in (module.lora_A.parameters() if hasattr(module.lora_A, "parameters") else [module.lora_A]):
                            if isinstance(p, (nn.Parameter, torch.Tensor)):
                                p.requires_grad = True
                    if hasattr(module, "lora_B") and module.lora_B is not None:
                        for p in (module.lora_B.parameters() if hasattr(module.lora_B, "parameters") else [module.lora_B]):
                            if isinstance(p, (nn.Parameter, torch.Tensor)):
                                p.requires_grad = True
                else:
                    replace_module(self.model, name, layer_compressed)
            except Exception as e:
                logger.warning("Module replacement failed for %s: %s", name, e)
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    total_comp_params += orig_layer_params
                    continue

            self.compressed_layers.append(name)
            comp_layer_params = layer_compressed.n_buckets
            if layer_compressed.lora_rank > 0:
                comp_layer_params += layer_compressed.lora_rank * (in_f + out_f)
            total_comp_params += comp_layer_params

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
                        (m for m in self.model.modules() if isinstance(m, (nn.Linear, HashedLinear))),
                        None,
                    )
                    in_dim = getattr(first_linear, "in_features", 8)
                    self.model(torch.randn(1, 8, in_dim, device=dev))
        except Exception as e:
            logger.warning("HashedNet post-compression forward check warning: %s", e)
            forward_check_success = False

        effective_ratio = total_comp_params / total_orig_params if total_orig_params > 0 else 1.0

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "fallback_layers": self.fallback_layers,
                "skipped_layers": self.skipped_layers,
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "num_skipped": len(self.skipped_layers),
                "layer_errors": self.layer_errors,
                "layer_buckets": self.layer_buckets,
                "max_error": self.max_error,
                "target_attention_only": self.target_attention_only,
                "fallback_to_original": self.fallback_to_original,
                "avg_reconstruction_error": avg_err,
                "n_buckets": self.base_buckets,
                "use_sign": self.use_sign,
                "lora_rank": self.lora_rank,
                "lora_alpha": self.lora_alpha,
                "effective_parameter_ratio": round(effective_ratio, 4),
                "forward_check_success": forward_check_success,
            }
        )
        logger.info(
            "HashedNet compression complete: %d compressed, %d fallback. Avg error: %.4f, Effective Param Ratio: %.2fx",
            len(self.compressed_layers),
            len(self.fallback_layers),
            avg_err,
            effective_ratio,
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("hashed_linear")
@COMPRESSOR_REGISTRY.register("hashed_linear_factory")
@COMPRESSOR_REGISTRY.register("hashed_net")
@COMPRESSOR_REGISTRY.register("hashed_net_compressor")
@COMPRESSOR_REGISTRY.register("hashed_net_factory")
@COMPRESSOR_REGISTRY.register("hashednets")
@COMPRESSOR_REGISTRY.register("hashednets_factory")
@COMPRESSOR_REGISTRY.register("hashednets_compressor")
def hashed_net_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[Union[int, float]] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for HashedNet compressor."""
    rank = kwargs.pop("rank", rank)
    n_buckets = kwargs.pop("n_buckets", _cfg_get(comp_config, "n_buckets", None))
    use_sign = kwargs.pop("use_sign", _cfg_get(comp_config, "use_sign", True))
    lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", _cfg_get(comp_config, "lora_r", 16)))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "hashed_net_max_error", _cfg_get(comp_config, "max_error", 0.35)))
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "hashed_net_target_attention_only", _cfg_get(comp_config, "target_attention_only", False)),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "hashed_net_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)),
    )
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = HashedNetCompressor(
        model,
        n_buckets=n_buckets,
        rank=rank,
        use_sign=use_sign,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        max_error=max_error,
        target_attention_only=target_attention_only,
        fallback_to_original=fallback,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


hashednets_factory = hashed_net_factory
HashedNetsCompressor = HashedNetCompressor

__all__ = ["HashedNetCompressor", "HashedNetsCompressor", "hashed_net_factory", "hashednets_factory"]
