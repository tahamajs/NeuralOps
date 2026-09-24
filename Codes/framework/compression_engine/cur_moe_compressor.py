"""
CUR-MoE Compressor: apply CUR cross-approximation to Mixture of Experts (MoE) modules and expert layers.
Registered in COMPRESSOR_REGISTRY as 'cur_moe' and 'cur_moe_compressor'.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import get_dynamic_compression_rank, should_compress_linear

from .cur_compressor import (
    CURLinear,
    compute_energy_scores,
    compute_leverage_scores,
    select_indices,
    tikhonov_pinv,
)

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract configuration attribute from dict, object, or namespace."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@COMPRESSOR_REGISTRY.register("cur_moe_class")
@COMPRESSOR_REGISTRY.register("cur_moe_compressor_class")
class CURMoECompressor(BaseCompressor):
    """
    Compresses Mixture of Experts (MoE) layers and dense FFN layers using CUR cross-approximation.

    Args:
        model: PyTorch model containing MoE or transformer layers.
        rank: Target rank for CUR decomposition.
        min_rank: Minimum allowable rank (default 8).
        method: 'leverage' (statistical leverage scores) or 'energy' (Frobenius energy).
        deterministic: Whether to use deterministic top-k selection (default True).
        max_error: Maximum allowed relative Frobenius error (default 0.90).
        fallback_to_original: Whether to preserve original layer if error > max_error (default True).
        max_retries: Number of adaptive rank escalations (default 2).
        target_attention_only: If True, skip attention projections and compress only FFN/MoE expert layers.
        cfg: Optional configuration object.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 64,
        min_rank: int = 8,
        method: str = "leverage",
        deterministic: bool = True,
        max_error: float = 0.90,
        fallback_to_original: bool = True,
        max_retries: int = 2,
        target_attention_only: bool = False,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="cur_moe", **kwargs)
        c_rank = _cfg_get(cfg, "cur_moe_rank", _cfg_get(cfg, "rank", rank))
        c_min_rank = _cfg_get(cfg, "cur_min_rank", _cfg_get(cfg, "min_rank", min_rank))
        c_method = _cfg_get(cfg, "cur_moe_method", _cfg_get(cfg, "method", method))
        c_det = _cfg_get(cfg, "cur_deterministic", _cfg_get(cfg, "deterministic", deterministic))
        c_max_error = _cfg_get(cfg, "cur_moe_max_error", _cfg_get(cfg, "max_error", max_error))
        c_fallback = _cfg_get(cfg, "fallback_to_original", fallback_to_original)
        c_target_attn = _cfg_get(cfg, "cur_moe_target_attention_only", _cfg_get(cfg, "target_attention_only", target_attention_only))

        self.rank = int(c_rank) if c_rank is not None else 64
        self.min_rank = int(c_min_rank) if c_min_rank is not None else 8
        self.method = str(c_method)
        self.deterministic = bool(c_det)
        self.max_error = float(c_max_error) if c_max_error is not None else 0.90
        self.fallback_to_original = bool(c_fallback)
        self.max_retries = max(0, int(max_retries))
        self.target_attention_only = bool(c_target_attn)
        self.cfg = cfg

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}
        self.total_orig_decomp_params = 0
        self.total_comp_decomp_params = 0

    def _compress_single_layer(self, name: str, module: nn.Module) -> bool:
        """Compress a single linear or Conv1D layer with Frobenius validation and adaptive retries."""
        target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
        cls_name = target_mod.__class__.__name__.lower()
        wrapper_name = module.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or "conv1d" in wrapper_name or hasattr(module, "nf") or hasattr(target_mod, "nf")
        has_lora = (hasattr(module, "lora_A") and module.lora_A is not None) or hasattr(module, "base_layer")

        raw_w = target_mod.weight.data
        orig_w = raw_w.t().float() if is_conv1d else raw_w.float()
        norm_orig = torch.norm(orig_w, p="fro").clamp_min(1e-8)
        if norm_orig.item() < 1e-4:
            logger.warning("CURMoECompressor: Layer %s weight Frobenius norm %.6f < 1e-4, skipping", name, norm_orig.item())
            return False

        out_f, in_f = orig_w.shape
        max_rank = min(out_f, in_f)

        if max_rank < self.min_rank:
            return False

        # Parameter reduction check: r*(in_f + out_f + r) < in_f * out_f
        target_r = min(self.rank, max_rank)
        while target_r >= self.min_rank and (target_r * (in_f + out_f + target_r)) >= 0.85 * (in_f * out_f):
            target_r = int(target_r * 0.8)

        if target_r < self.min_rank:
            return False

        cur_r = target_r
        best_cur_layer = None
        best_err = float("inf")
        best_r = cur_r

        for _ in range(self.max_retries + 1):
            try:
                candidate = CURLinear.from_linear(
                    target_mod,
                    rank=cur_r,
                    method=self.method,
                    deterministic=self.deterministic,
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
                logger.warning("CURMoECompressor: Error on %s with rank %d: %s", name, cur_r, ex)
                break

        if best_err > self.max_error or best_cur_layer is None:
            logger.warning(
                "CURMoECompressor: Layer %s error %.4f exceeded max_error %.4f",
                name, best_err, self.max_error
            )
            if self.fallback_to_original:
                self.fallback_layers.append(name)
                # Freeze fallback layers
                for p in module.parameters():
                    p.requires_grad = False
                return False

        new_layer = best_cur_layer
        compressed_layer = new_layer

        has_base_layer = hasattr(module, 'base_layer') or (hasattr(module, 'linear') and (hasattr(module, 'lora_A') or hasattr(module, 'r')))
        # Preserve existing LoRA adapter weights if the source module has them and not wrapped
        if not has_base_layer and hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and \
           module.lora_A is not None and module.lora_B is not None:
            src_lA = getattr(module.lora_A, 'weight', module.lora_A)
            src_lB = getattr(module.lora_B, 'weight', module.lora_B)
            dst_lA = getattr(new_layer, 'lora_A', None)
            dst_lB = getattr(new_layer, 'lora_B', None)
            if dst_lA is not None and hasattr(src_lA, 'data') and hasattr(dst_lA, 'data') and src_lA.shape == dst_lA.shape:
                dst_lA.data.copy_(src_lA.data)
            if dst_lB is not None and hasattr(src_lB, 'data') and hasattr(dst_lB, 'data') and src_lB.shape == dst_lB.shape:
                dst_lB.data.copy_(src_lB.data)

        if hasattr(module, 'base_layer'):
            module.base_layer = compressed_layer  # put compressed layer inside wrapper
            new_module = module
        elif hasattr(module, 'linear') and (hasattr(module, 'lora_A') or hasattr(module, 'r')):
            module.linear = compressed_layer  # put compressed layer inside framework LoRALinear
            new_module = module
        else:
            new_module = compressed_layer

        if has_base_layer:
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
        self.total_orig_decomp_params += (in_f * out_f)
        self.total_comp_decomp_params += (best_r * (in_f + out_f + best_r))
        return True

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress expert linear layers in MoE architecture, or fallback to FFN layers."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        self.total_orig_decomp_params = 0
        self.total_comp_decomp_params = 0
        experts_detected = False

        # Scan for explicit MoE modules (module.experts or expert submodules)
        for name, module in list(self.model.named_modules()):
            name_lower = name.lower()
            if any(k in name_lower for k in ["router", "gate", "classifier", "score", "lm_head", "pooler", "embedding"]):
                continue

            if hasattr(module, "experts"):
                experts_container = module.experts
                if isinstance(experts_container, (nn.ModuleList, list)):
                    experts_detected = True
                    for exp_idx, expert in enumerate(experts_container):
                        for child_name, child_mod in list(expert.named_children()):
                            child_full_name = f"{name}.experts.{exp_idx}.{child_name}" if name else f"experts.{exp_idx}.{child_name}"
                            is_linear_like = (
                                isinstance(child_mod, nn.Linear)
                                or "conv1d" in child_mod.__class__.__name__.lower()
                                or hasattr(child_mod, "nf")
                                or hasattr(child_mod, "base_layer")
                            )
                            if is_linear_like and not isinstance(child_mod, CURLinear):
                                self._compress_single_layer(child_full_name, child_mod)
            elif "expert" in name_lower:
                is_linear_like = (
                    isinstance(module, nn.Linear)
                    or "conv1d" in module.__class__.__name__.lower()
                    or hasattr(module, "nf")
                    or hasattr(module, "base_layer")
                )
                if is_linear_like and not isinstance(module, CURLinear):
                    experts_detected = True
                    self._compress_single_layer(name, module)

        # Fallback to FFN linear layers if no explicit MoE structure was found
        _ATTN_NAMES = frozenset({
            "query", "key", "value",
            "q_lin", "k_lin", "v_lin",
            "q_proj", "k_proj", "v_proj",
            "out_lin", "out_proj", "o_proj",
            "c_attn",
        })
        if len(self.compressed_layers) == 0:
            visited = set()
            for name, module in list(self.model.named_modules()):
                target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
                if id(target_mod) in visited:
                    continue
                visited.add(id(target_mod))
                name_lower = name.lower()
                if any(skip in name_lower for skip in ["classifier", "score", "lm_head", "pooler", "embedding", "router", "gate"]):
                    continue

                last_part = name.split(".")[-1]
                is_attn_proj = (last_part in _ATTN_NAMES) or any(k in name_lower for k in ("attn", "attention"))

                # In target_attention_only mode (or default MoE behavior), we protect attention projections
                if is_attn_proj:
                    if self.target_attention_only:
                        # Freeze attention layers to protect them during fine-tuning
                        for p in module.parameters():
                            p.requires_grad = False
                    continue

                is_linear_like = (
                    isinstance(module, nn.Linear)
                    or "conv1d" in module.__class__.__name__.lower()
                    or hasattr(module, "nf")
                    or hasattr(module, "base_layer")
                )
                if is_linear_like and not isinstance(module, CURLinear):
                    self._compress_single_layer(name, module)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (sum(self.layer_errors.values()) / len(self.layer_errors)) if self.layer_errors else 0.0
        effective_param_ratio = (
            self.total_comp_decomp_params / self.total_orig_decomp_params if self.total_orig_decomp_params > 0 else 1.0
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
                "cur_moe_rank": self.rank,
                "min_rank": self.min_rank,
                "deterministic": self.deterministic,
                "experts_detected": experts_detected,
                "target_attention_only": self.target_attention_only,
                "effective_parameter_ratio": round(effective_param_ratio, 4),
            }
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("cur_moe")
@COMPRESSOR_REGISTRY.register("cur_moe_compressor")
@COMPRESSOR_REGISTRY.register("cur_moe_factory")
@COMPRESSOR_REGISTRY.register("cur_moe_compressor_factory")
def cur_moe_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for CUR-MoE compressor."""
    effective_rank = kwargs.pop("rank", rank or _cfg_get(comp_config, "cur_moe_rank", _cfg_get(comp_config, "rank", 64)))
    min_rank = kwargs.pop("min_rank", _cfg_get(comp_config, "min_rank", 8))
    method = kwargs.pop("method", _cfg_get(comp_config, "cur_moe_method", _cfg_get(comp_config, "method", "leverage")))
    deterministic = kwargs.pop("deterministic", _cfg_get(comp_config, "deterministic", True))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "cur_moe_max_error", _cfg_get(comp_config, "max_error", 0.90)))
    fallback = kwargs.pop("fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True))
    target_attention_only = kwargs.pop("target_attention_only", _cfg_get(comp_config, "cur_moe_target_attention_only", _cfg_get(comp_config, "target_attention_only", False)))
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = CURMoECompressor(
        model,
        rank=effective_rank,
        min_rank=min_rank,
        method=method,
        deterministic=deterministic,
        max_error=max_error,
        fallback_to_original=fallback,
        target_attention_only=target_attention_only,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


def inject_cur_moe(model: nn.Module, **kwargs: Any) -> Optional[nn.Module]:
    """Safe injection hook for cur_moe. Catches exceptions, logs, and returns None on failure."""
    try:
        res = cur_moe_factory(model, **kwargs)
        if isinstance(res, tuple):
            return res[0]
        return res
    except Exception as e:
        logger.error(f"Failed to inject cur_moe: {e}", exc_info=True)
        return None


def cur_moe_inject(model: nn.Module, **kwargs: Any) -> Optional[nn.Module]:
    """Safe injection hook for cur_moe alias."""
    return inject_cur_moe(model, **kwargs)
