"""
CUR Compressor: replaces Linear and Conv1D layers with CUR decomposition.
Uses leverage score sampling or energy sampling for column/row selection.
Registered in COMPRESSOR_REGISTRY as 'cur' and 'cur_compressor'.
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

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    Conv1D = None

try:
    from .cur_linear import CURLinear
except ImportError:
    try:
        from framework.compression_engine.cur_linear import CURLinear
    except ImportError:
        from cur_linear import CURLinear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract configuration attribute from dict, object, or namespace."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@COMPRESSOR_REGISTRY.register("cur_compressor")
class CURCompressor(BaseCompressor):
    """
    Compressor that replaces eligible Linear and Conv1D layers with CURLinear.

    Args:
        model: PyTorch model.
        rank: Target rank for skeleton column and row selection (default 64).
        min_rank: Minimum allowable rank for compression (default 8).
        method: Selection method ('leverage' or 'energy', default 'leverage').
        deterministic: Deterministic top-k selection if True (default True).
        max_error: Maximum relative Frobenius reconstruction error allowed (default 0.20).
        fallback_to_original: If True, retains original layer when reconstruction error exceeds max_error (default True).
        max_retries: Number of adaptive rank retries when error threshold is exceeded (default 2).
        lora_r: LoRA adapter rank if adding fine-tuning adapters (default 0).
        lora_alpha: LoRA scaling factor (default 1.0).
        freeze_skeletons: If True, freeze C and R parameters during post-compression training (default False).
        target_attention_only: If True, restrict compression to attention projections only.
        target_names: Optional list of layer name substrings to target.
        exclude_names: Optional list of layer name substrings to exclude.
        cfg: Optional configuration object.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 64,
        min_rank: int = 8,
        method: str = "leverage",
        deterministic: bool = True,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        max_retries: int = 2,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        freeze_skeletons: bool = False,
        target_attention_only: bool = False,
        target_names: Optional[List[str]] = None,
        exclude_names: Optional[List[str]] = None,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="cur", **kwargs)
        c_rank = _cfg_get(cfg, "cur_rank", _cfg_get(cfg, "rank", rank))
        c_min_rank = _cfg_get(cfg, "cur_min_rank", _cfg_get(cfg, "min_rank", min_rank))
        c_method = _cfg_get(cfg, "cur_method", _cfg_get(cfg, "method", method))
        c_det = _cfg_get(cfg, "cur_deterministic", _cfg_get(cfg, "deterministic", deterministic))
        c_max_error = _cfg_get(cfg, "cur_max_error", _cfg_get(cfg, "max_error", max_error))
        c_fallback = _cfg_get(cfg, "fallback_to_original", fallback_to_original)
        c_lora_r = _cfg_get(cfg, "cur_lora_r", _cfg_get(cfg, "lora_r", lora_r))
        c_lora_alpha = _cfg_get(cfg, "cur_lora_alpha", _cfg_get(cfg, "lora_alpha", lora_alpha))
        c_freeze = _cfg_get(cfg, "freeze_skeletons", freeze_skeletons)
        c_target_attn = _cfg_get(cfg, "cur_target_attention_only", _cfg_get(cfg, "target_attention_only", target_attention_only))

        self.rank = int(c_rank) if c_rank is not None else 64
        self.min_rank = int(c_min_rank) if c_min_rank is not None else 8
        self.method = str(c_method)
        self.deterministic = bool(c_det)
        self.max_error = float(c_max_error) if c_max_error is not None else 0.20
        self.fallback_to_original = bool(c_fallback)
        self.max_retries = max(0, int(max_retries))
        self.lora_r = int(c_lora_r or 0)
        self.lora_alpha = float(c_lora_alpha or 1.0)
        self.freeze_skeletons = bool(c_freeze)
        self.target_attention_only = bool(c_target_attn)
        self.cfg = cfg
        self.target_names = target_names or _cfg_get(cfg, "cur_target_names", _cfg_get(cfg, "target_layers", None))
        self.exclude_names = exclude_names or _cfg_get(cfg, "cur_exclude_names", [
            "lm_head", "classifier", "head", "score", "pooler", "embedding", "router", "gate"
        ])

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """Identify candidate linear/Conv1D/LoRALinear layers."""
        cls_name = module.__class__.__name__.lower()
        has_base = hasattr(module, "base_layer")
        base_cls = module.base_layer.__class__.__name__.lower() if has_base else ""

        is_linear = isinstance(module, nn.Linear) or "linear" in cls_name
        is_conv1d = "conv1d" in cls_name or hasattr(module, "nf")
        is_lora = has_base and (
            isinstance(module.base_layer, nn.Linear)
            or "conv1d" in base_cls
            or hasattr(module.base_layer, "nf")
        )

        if not (is_linear or is_conv1d or is_lora):
            return False

        if isinstance(module, CURLinear) or (has_base and isinstance(module.base_layer, CURLinear)):
            return False

        name_lower = name.lower()
        if any(skip in name_lower for skip in self.exclude_names):
            return False

        if self.target_attention_only:
            ffn_patterns = [
                "ffn", "mlp", "lin1", "lin2", "intermediate",
                "gate_proj", "up_proj", "down_proj", "c_fc"
            ]
            if any(p in name_lower for p in ffn_patterns):
                return False
            if "output.dense" in name_lower and "attention" not in name_lower:
                return False
            attn_patterns = [
                "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
                "out_lin", "q_lin", "k_lin", "v_lin", "c_attn", "c_proj",
                "query", "key", "value", "in_proj_qkv", "out_proj"
            ]
            if not any(p in name_lower for p in attn_patterns):
                return False

        if self.target_names and not any(t in name for t in self.target_names):
            return False

        return True

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Replace eligible Linear and Conv1D layers with CURLinear."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        visited_weight_ids: Set[int] = set()

        total_orig_decomp_params = 0
        total_comp_decomp_params = 0

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                if self.target_attention_only:
                    is_lin_like = isinstance(module, nn.Linear) or "conv1d" in module.__class__.__name__.lower() or hasattr(module, "base_layer")
                    if is_lin_like:
                        for p in module.parameters():
                            p.requires_grad = False
                continue

            target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
            cls_name = module.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(module, "nf") or hasattr(target_mod, "nf")
            has_lora = (hasattr(module, "lora_A") and module.lora_A is not None) or hasattr(module, "base_layer")

            # Check tied weights
            if not hasattr(target_mod, "weight") or target_mod.weight is None:
                for p in module.parameters():
                    p.requires_grad = False
                continue
            weight_id = id(target_mod.weight)
            if weight_id in visited_weight_ids:
                logger.info("CURCompressor: Skipping tied weight layer %s", name)
                for p in module.parameters():
                    p.requires_grad = False
                continue

            raw_w = target_mod.weight.data
            orig_w = raw_w.t().float() if is_conv1d else raw_w.float()
            out_f, in_f = orig_w.shape
            max_rank = min(out_f, in_f)

            if max_rank < self.min_rank:
                for p in module.parameters():
                    p.requires_grad = False
                continue

            # Parameter reduction check: r*(in_f + out_f + r) < in_f * out_f
            target_r = min(self.rank, max_rank)
            while target_r >= self.min_rank and (target_r * (in_f + out_f + target_r)) >= 0.85 * (in_f * out_f):
                target_r = int(target_r * 0.8)

            if target_r < self.min_rank:
                logger.debug("CURCompressor: Skipping %s as target rank %d violates parameter reduction constraint", name, target_r)
                for p in module.parameters():
                    p.requires_grad = False
                continue

            cur_r = target_r
            best_cur_layer = None
            best_err = float("inf")
            best_r = cur_r
            norm_orig = torch.norm(orig_w, p="fro").clamp_min(1e-8)
            effective_lora_r = 0 if has_lora else self.lora_r

            for _ in range(self.max_retries + 1):
                try:
                    candidate = CURLinear.from_linear(
                        target_mod,
                        rank=cur_r,
                        method=self.method,
                        deterministic=self.deterministic,
                        lora_r=effective_lora_r,
                        lora_alpha=self.lora_alpha,
                        freeze_skeletons=self.freeze_skeletons,
                    )
                    recon_w = candidate.get_weight().float()
                    err = (torch.norm(orig_w - recon_w, p="fro") / norm_orig).item()

                    if err < best_err:
                        best_err = err
                        best_cur_layer = candidate
                        best_r = cur_r

                    if err <= self.max_error:
                        break

                    next_r = min(int(cur_r * 1.5), max_rank - 1)
                    if next_r <= cur_r or (next_r * (in_f + out_f + next_r)) >= 0.85 * (in_f * out_f):
                        break
                    cur_r = next_r
                except Exception as ex:
                    logger.warning("CURCompressor: Failed CUR decomposition on %s at rank %d: %s", name, cur_r, ex)
                    break

            if best_err > self.max_error:
                logger.warning(
                    "CURCompressor: Layer %s error %.4f exceeded max_error %.4f",
                    name, best_err, self.max_error
                )
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    # Freeze fallback layers
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

            if best_cur_layer is not None:
                new_layer = best_cur_layer
                compressed_layer = new_layer
                # Preserve existing LoRA adapter weights if the source module has them
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and \
                   module.lora_A is not None and module.lora_B is not None:
                    src_lA = getattr(module.lora_A, 'weight', module.lora_A)
                    src_lB = getattr(module.lora_B, 'weight', module.lora_B)
                    if hasattr(new_layer, 'lora_A') and new_layer.lora_A is not None:
                        if hasattr(src_lA, 'data') and src_lA.shape == new_layer.lora_A.shape:
                            new_layer.lora_A.data.copy_(src_lA.data)
                    if hasattr(new_layer, 'lora_B') and new_layer.lora_B is not None:
                        if hasattr(src_lB, 'data') and src_lB.shape == new_layer.lora_B.shape:
                            new_layer.lora_B.data.copy_(src_lB.data)

                if hasattr(module, 'base_layer'):
                    module.base_layer = compressed_layer  # put compressed layer inside wrapper
                    # module itself (LoRALinear wrapper) remains the replacement
                    new_module = module
                else:
                    new_module = compressed_layer

                if hasattr(module, 'base_layer'):
                    for p_name, p in module.named_parameters():
                        if any(t in p_name.lower() for t in ("lora_a", "lora_b", "adapter")):
                            p.requires_grad = True
                else:
                    if name == "":
                        self.model = new_module
                    else:
                        replace_module(self.model, name, new_module)

                self.compressed_layers.append(name)
                self.layer_errors[name] = float(best_err)
                self.layer_ranks[name] = int(best_r)
                visited_weight_ids.add(weight_id)

                total_orig_decomp_params += (in_f * out_f)
                total_comp_decomp_params += (best_r * (in_f + out_f + best_r))

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (sum(self.layer_errors.values()) / len(self.layer_errors)) if self.layer_errors else 0.0
        effective_param_ratio = (
            total_comp_decomp_params / total_orig_decomp_params if total_orig_decomp_params > 0 else 1.0
        )
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "compressed_layer_count": len(self.compressed_layers),
                "fallback_layers": self.fallback_layers,
                "fallback_layer_count": len(self.fallback_layers),
                "layer_errors": self.layer_errors,
                "layer_ranks": self.layer_ranks,
                "avg_reconstruction_error": avg_err,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "cur_method": self.method,
                "cur_rank": self.rank,
                "min_rank": self.min_rank,
                "deterministic": self.deterministic,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "target_attention_only": self.target_attention_only,
                "effective_parameter_ratio": round(effective_param_ratio, 4),
            }
        )
        return self.model, stats

    def compress_linear(self, linear: Union[nn.Linear, Any]) -> CURLinear:
        """Compress a single linear layer into CURLinear."""
        return CURLinear.from_linear(
            linear,
            rank=self.rank,
            method=self.method,
            deterministic=self.deterministic,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            freeze_skeletons=self.freeze_skeletons,
        )


@COMPRESSOR_REGISTRY.register("cur")
@COMPRESSOR_REGISTRY.register("cur_linear")
@COMPRESSOR_REGISTRY.register("cure")
@COMPRESSOR_REGISTRY.register("cur_factory")
@COMPRESSOR_REGISTRY.register("cur_linear_factory")
@COMPRESSOR_REGISTRY.register("cure_factory")
@COMPRESSOR_REGISTRY.register("cur_compressor_factory")
def cur_compressor_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for CUR compression."""
    effective_rank = kwargs.pop("rank", rank or _cfg_get(comp_config, "cur_rank", _cfg_get(comp_config, "rank", 64)))
    min_rank = kwargs.pop("min_rank", _cfg_get(comp_config, "min_rank", 8))
    method = kwargs.pop("method", _cfg_get(comp_config, "cur_method", _cfg_get(comp_config, "method", "leverage")))
    deterministic = kwargs.pop("deterministic", _cfg_get(comp_config, "deterministic", True))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "cur_max_error", _cfg_get(comp_config, "max_error", 0.20)))
    fallback = kwargs.pop("fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True))
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "cur_lora_r", _cfg_get(comp_config, "lora_r", 0)))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "cur_lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0)))
    target_attention_only = kwargs.pop("target_attention_only", _cfg_get(comp_config, "cur_target_attention_only", _cfg_get(comp_config, "target_attention_only", False)))
    target_names = kwargs.pop("target_names", _cfg_get(comp_config, "cur_target_names", None))
    exclude_names = kwargs.pop("exclude_names", _cfg_get(comp_config, "cur_exclude_names", None))
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = CURCompressor(
        model,
        rank=effective_rank,
        min_rank=min_rank,
        method=method,
        deterministic=deterministic,
        max_error=max_error,
        fallback_to_original=fallback,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_attention_only=target_attention_only,
        target_names=target_names,
        exclude_names=exclude_names,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model
