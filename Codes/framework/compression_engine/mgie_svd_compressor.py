"""
MGIE-SVD with Total Rank Budget: Global Entropy-Proportional Rank Allocation.
Registered as 'mgie_budget'.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_ATTN_NAMES = {
    "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
    "c_attn", "c_proj", "query", "key", "value", "q_lin", "k_lin", "v_lin", "out_lin", "self_attn"
}


@COMPRESSOR_REGISTRY.register("mgie_budget")
class MGIESVDBudgetCompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    MGIE-SVD Budget Compressor: Global Entropy-Proportional Rank Allocation.
    """

    def __init__(
        self,
        model: nn.Module,
        total_rank_budget: int = 1000,
        rank: Optional[int] = None,
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = False,
        target_attention_only: bool = False,
        min_rank: int = 4,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="mgie_budget", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            total_rank_budget = _cfg_get(
                cfg, "total_rank_budget", _cfg_get(cfg, "rank_budget", total_rank_budget)
            )
            rank = _cfg_get(cfg, "mgie_rank", _cfg_get(cfg, "rank", rank))
            max_error = _cfg_get(cfg, "mgie_max_error", _cfg_get(cfg, "max_error", max_error))
            fallback_to_original = _cfg_get(
                cfg, "mgie_fallback_to_original", _cfg_get(cfg, "fallback_to_original", fallback_to_original)
            )
            lora_r = _cfg_get(cfg, "lora_r", lora_r)
            lora_alpha = _cfg_get(cfg, "lora_alpha", lora_alpha)
            skip_attention_qkv = _cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv)
            target_attention_only = _cfg_get(cfg, "target_attention_only", target_attention_only)
            min_rank = _cfg_get(cfg, "min_rank", min_rank)

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]

        self.total_rank_budget = int(total_rank_budget)
        self.rank = rank
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.target_attention_only = bool(target_attention_only)
        self.min_rank = max(1, int(min_rank))
        self.target_modules = target_modules

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}
        self.stats: Dict[str, Any] = {}

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
        cls_name = module.__class__.__name__.lower()
        is_linear_like = (
            isinstance(module, nn.Linear)
            or "conv1d" in cls_name
            or hasattr(module, "nf")
            or hasattr(module, "base_layer")
        )
        if not is_linear_like:
            return False
        if isinstance(module, SVDLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self._is_gqa_asymmetric(name, module):
            return False
        if self.target_attention_only:
            is_attn = any(k in name.lower() for k in _ATTN_NAMES)
            if not is_attn:
                return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    @staticmethod
    def _compute_entropy(W: torch.Tensor) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute singular entropy on squared singular values (energy distribution),
        normalized by layer parameter count sqrt(d_in * d_out), returning SVD components in FP32."""
        d_out, d_in = W.shape
        U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
        energy = S ** 2
        total_energy = energy.sum()
        if total_energy <= 1e-12:
            return 0.0, U, S, Vt
        p = torch.clamp(energy / total_energy, min=1e-12, max=1.0)
        entropy = -torch.sum(p * torch.log(p)).item()
        norm_factor = math.sqrt(float(d_in * d_out))
        normalized_entropy = entropy / max(1e-6, norm_factor)
        return max(0.0, float(normalized_entropy)), U, S, Vt

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress all linear layers according to global entropy budget."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        seen_weights: Set[int] = set()
        seen_param_ids: Set[int] = set()

        total_orig_params = sum(p.numel() for p in self.model.parameters())
        total_comp_params = 0

        def _add_params(mod: nn.Module):
            nonlocal total_comp_params
            for p in mod.parameters(recurse=False):
                if id(p) not in seen_param_ids:
                    seen_param_ids.add(id(p))
                    total_comp_params += p.numel()

        target_layers = []
        for name, mod in list(self.model.named_modules()):
            if self._is_target_module(name, mod):
                target_layer = getattr(mod, "base_layer", mod)
                if not hasattr(target_layer, "weight") or target_layer.weight is None:
                    _add_params(mod)
                    continue
                w_id = id(target_layer.weight)
                if w_id in seen_weights:
                    self.skipped_layers.append(name)
                    _add_params(mod)
                    continue
                seen_weights.add(w_id)
                target_layers.append((name, mod, target_layer))
            else:
                _add_params(mod)

        if not target_layers:
            logger.warning("MGIE-Budget: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        # Step 1: Collect singular entropies and SVD factors
        layer_data = {}
        total_entropy = 0.0
        for name, module, target_layer in target_layers:
            raw_w = target_layer.weight.data
            cls_name = target_layer.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(target_layer, "nf")
            W = raw_w.t().contiguous().clone().float() if is_conv1d else raw_w.clone().float()

            ent, U, S, Vt = self._compute_entropy(W)
            layer_data[name] = (module, target_layer, ent, U, S, Vt, W, is_conv1d)
            total_entropy += max(ent, 1e-4)

        # Step 2: Allocate rank budgets and factorize
        for name, (module, target_layer, ent, U, S, Vt, W, is_conv1d) in layer_data.items():
            d_out, d_in = W.shape
            max_r = min(d_out, d_in)
            if self.rank is not None:
                r = min(self.rank, max_r)
            else:
                frac = max(ent, 1e-4) / total_entropy
                r = int(round(frac * self.total_rank_budget))
            r = max(min(self.min_rank, max_r), min(r, max_r))
            r = min(r, len(S))

            device = target_layer.weight.device
            dtype = target_layer.weight.dtype

            try:
                sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
                V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(dtype=dtype, device=device)

                W_approx = torch.nan_to_num(U_r @ V_r, nan=0.0)
                w_norm = torch.norm(W, p="fro").item()
                err = (torch.norm(W.to(device) - W_approx, p="fro") / max(w_norm, 1e-12)).item()
                self.layer_errors[name] = float(err)
                self.layer_ranks[name] = r

                if not math.isfinite(err) or err > self.max_error:
                    logger.warning(
                        "MGIE-Budget reconstruction error %.4f on %s exceeded threshold %.4f.",
                        err, name, self.max_error,
                    )
                    if self.fallback_to_original:
                        self.fallback_layers.append(name)
                        for p in module.parameters(recurse=False):
                            p.requires_grad = False
                        _add_params(module)
                        continue

                existing_lora_r = getattr(module, "r", getattr(module, "lora_r", 0))
                eff_lora_r = existing_lora_r if existing_lora_r > 0 else self.lora_r
                eff_lora_alpha = getattr(module, "lora_alpha", None) or getattr(module, "alpha", None) or self.lora_alpha

                svd_layer = SVDLinear(
                    in_features=d_in,
                    out_features=d_out,
                    rank=r,
                    bias=target_layer.bias is not None,
                    dtype=dtype,
                    device=device,
                    lora_r=eff_lora_r,
                    lora_alpha=eff_lora_alpha,
                )
                svd_layer.set_low_rank_factors(U_r, V_r)
                if target_layer.bias is not None:
                    svd_layer.bias.data.copy_(target_layer.bias.data)

                # Preserve pre-existing LoRA adapter parameters if wrapped
                if hasattr(module, "lora_A") and hasattr(module, "lora_B") and svd_layer.lora_A is not None and svd_layer.lora_B is not None:
                    with torch.no_grad():
                        src_A = module.lora_A.weight.data if hasattr(module.lora_A, "weight") else module.lora_A.data
                        src_B = module.lora_B.weight.data if hasattr(module.lora_B, "weight") else module.lora_B.data
                        if src_A.shape == svd_layer.lora_A.shape and src_B.shape == svd_layer.lora_B.shape:
                            svd_layer.lora_A.copy_(src_A.to(svd_layer.lora_A.device, dtype=svd_layer.lora_A.dtype))
                            svd_layer.lora_B.copy_(src_B.to(svd_layer.lora_B.device, dtype=svd_layer.lora_B.dtype))

                if svd_layer.lora_A is not None:
                    svd_layer.lora_A.requires_grad = True
                if svd_layer.lora_B is not None:
                    svd_layer.lora_B.requires_grad = True

                replace_module(self.model, name, svd_layer)
                self.compressed_layers.append(name)
                _add_params(svd_layer)
            except Exception as e:
                logger.warning("MGIE-Budget compression exception on %s: %s", name, e)
                self.fallback_layers.append(name)
                self.layer_errors[name] = 1.0
                for p in module.parameters(recurse=False):
                    p.requires_grad = False
                _add_params(module)

        # Post-compression forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev)
                try:
                    self.model(dummy)
                except Exception:
                    self.model(dummy.float())
        except Exception as e:
            logger.warning("MGIE-Budget post-compression forward check warning: %s", e)
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        eff_ratio = total_comp_params / max(1, total_orig_params)
        stats = self.get_compression_stats()
        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "skipped_layers": self.skipped_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            "layer_errors": self.layer_errors,
            "layer_ranks": self.layer_ranks,
            "avg_reconstruction_error": avg_err,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "effective_parameter_ratio": eff_ratio,
            "total_rank_budget": self.total_rank_budget,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "forward_check_success": forward_check_success,
            "method": "mgie_budget",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("mgie_budget_factory")
@COMPRESSOR_REGISTRY.register("mgie_svd_budget_factory")
def mgie_budget_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """MGIE budget factory function."""
    total_rank_budget = kwargs.pop(
        "total_rank_budget",
        _cfg_get(comp_config, "total_rank_budget", _cfg_get(comp_config, "rank_budget", 1000)),
    )
    rank = kwargs.pop("rank", _cfg_get(comp_config, "mgie_rank", _cfg_get(comp_config, "rank", None)))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "mgie_max_error", _cfg_get(comp_config, "max_error", 0.50)))
    fallback = kwargs.pop("fallback_to_original", _cfg_get(
        comp_config, "mgie_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)
    ))
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    target_attention_only = kwargs.pop("target_attention_only", _cfg_get(comp_config, "target_attention_only", False))
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = MGIESVDBudgetCompressor(
        model,
        total_rank_budget=total_rank_budget,
        rank=rank,
        max_error=max_error,
        fallback_to_original=fallback,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_attention_only=target_attention_only,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "MGIESVDBudgetCompressor",
    "mgie_budget_factory",
]
