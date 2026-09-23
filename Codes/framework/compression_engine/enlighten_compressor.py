"""
ENLighten: Low-Rank SVD + 2:4 Structured Sparsity (Photonic-Aligned).
Registered as 'enlighten' and 'enlighten_compressor'.

Bug-fix map (2026 revision):
  EN-FIX-01  is_custom_attention = False on compressor (it's a compressor, not attention).
  EN-FIX-02  Factory registered under 'enlighten_factory'; class under 'enlighten_class'.
             Factory still exposed as 'enlighten' for framework compatibility.
  EN-FIX-03  Debug logging in _is_target_module when no targets found.
  EN-FIX-04  skip_should_compress_filter bypass option.
  EN-FIX-05  Single source of truth: _is_target_module used in compress() too.
  EN-FIX-06  layer_ranks tracks the rank actually used in best_w_new.
  EN-FIX-07  Explicit effective_parameter_ratio in stats (real metric of sparsification).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
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
    "q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj",
    "query", "key", "value",
    "q_lin", "k_lin", "v_lin", "out_lin",
}


def vectorized_24_prune(residual: torch.Tensor) -> torch.Tensor:
    """2:4 structured sparsity along groups of 4 input columns."""
    out_f, in_f = residual.shape
    rem = in_f % 4
    if rem != 0:
        residual_padded = F.pad(residual, (0, 4 - rem), mode="constant", value=0.0)
    else:
        residual_padded = residual
    num_blocks = residual_padded.shape[1] // 4
    blocks = residual_padded.view(out_f, num_blocks, 4)
    _, top_idx = torch.topk(blocks.abs(), k=2, dim=-1)
    mask = torch.zeros_like(blocks)
    mask.scatter_(-1, top_idx, 1.0)
    pruned = (blocks * mask).view(out_f, -1)
    if rem != 0:
        return pruned[:, :in_f]
    return pruned


def unstructured_prune(residual: torch.Tensor, sparsity: float) -> torch.Tensor:
    if sparsity <= 0.0:
        return residual
    if sparsity >= 1.0:
        return torch.zeros_like(residual)
    k_prune = int(residual.numel() * sparsity)
    if k_prune <= 0:
        return residual
    flat = residual.abs().flatten()
    threshold = torch.kthvalue(flat, k_prune).values
    mask = (residual.abs() > threshold).to(residual.dtype)
    return residual * mask


# ===========================================================================
# Compressor class (registered as *_class to avoid factory name collision)
# ===========================================================================

@COMPRESSOR_REGISTRY.register("enlighten_class")
@COMPRESSOR_REGISTRY.register("enlighten_compressor_class")
class ENLightenCompressor(BaseCompressor):
    # EN-FIX-01: this is a compressor, not an attention layer.
    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module,
        rank: int = 64,
        sparsity: float = 0.5,
        structured_24: bool = True,
        max_error: float = 0.35,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        max_retries: int = 2,
        target_modules: Optional[List[str]] = None,
        min_layer_dim: int = 8,
        skip_should_compress_filter: bool = False,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="enlighten", **kwargs)
        cfg = cfg or kwargs.get("comp_config", None)
        self.cfg = cfg

        self.rank = int(_cfg_get(cfg, "enlighten_rank", _cfg_get(cfg, "rank", rank)))
        self.sparsity = float(_cfg_get(cfg, "enlighten_sparsity",
                                       _cfg_get(cfg, "sparsity", sparsity)))
        self.structured_24 = bool(_cfg_get(cfg, "enlighten_structured_24",
                                           _cfg_get(cfg, "structured_24", structured_24)))
        self.max_error = float(_cfg_get(cfg, "enlighten_max_error",
                                        _cfg_get(cfg, "max_error", max_error)))
        self.target_attention_only = bool(_cfg_get(
            cfg, "enlighten_target_attention_only",
            _cfg_get(cfg, "target_attention_only", target_attention_only),
        ))
        self.fallback_to_original = bool(_cfg_get(
            cfg, "enlighten_fallback_to_original",
            _cfg_get(cfg, "fallback_to_original", fallback_to_original),
        ))
        self.max_retries = max(0, int(max_retries))
        self.target_modules = target_modules
        self.min_layer_dim = max(1, int(min_layer_dim))
        self.skip_should_compress_filter = bool(skip_should_compress_filter)

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_ranks: Dict[str, int] = {}
        self.layer_sparsities: Dict[str, float] = {}
        self.layer_errors: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}
        self._compressed_params: int = 0

    # ------------------------------------------------------------------
    # EN-FIX-03, EN-FIX-05: centralized target check
    # ------------------------------------------------------------------
    def _is_target_module(self, name: str, module: nn.Module, _debug: bool = False) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
            if _debug:
                logger.info(f"  [skip] {name}: lora/base_layer part")
            return False
        if any(skip in name_lower for skip in
               ["classifier", "score", "lm_head", "pooler", "embedding", "router", "gate"]):
            if _debug:
                logger.info(f"  [skip] {name}: skip keyword")
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        if not (is_linear or is_conv1d):
            if _debug:
                logger.info(f"  [skip] {name}: not Linear/Conv1D ({cls_name})")
            return False

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

        w = getattr(target_mod, "weight", None)
        if w is None:
            if _debug:
                logger.info(f"  [skip] {name}: weight None")
            return False
        if min(w.shape[0], w.shape[1]) < self.min_layer_dim:
            if _debug:
                logger.info(f"  [skip] {name}: dim too small {tuple(w.shape)}")
            return False

        if self.target_attention_only:
            last_part = name.split(".")[-1]
            is_attn = (last_part in _ATTN_NAMES) or any(
                k in name_lower for k in ("attn", "attention"))
            if not is_attn:
                if _debug:
                    logger.info(f"  [skip] {name}: target_attention_only filter")
                return False

        if self.target_modules is not None:
            ok = any(t in name for t in self.target_modules)
            if not ok and _debug:
                logger.info(f"  [skip] {name}: not in target_modules")
            return ok

        if _debug:
            logger.info(f"  [TARGET] {name}: shape={tuple(w.shape)}")
        return True

    # ------------------------------------------------------------------
    # Decompose
    # ------------------------------------------------------------------
    def _decompose_and_prune(
        self, W: torch.Tensor, rank: int,
    ) -> Tuple[torch.Tensor, float]:
        out_f, in_f = W.shape
        r = max(1, min(rank, out_f, in_f))
        norm_w = torch.linalg.norm(W, ord="fro").item()
        if norm_w <= 1e-12:
            return W.clone(), 0.0

        U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        U_r = U[:, :r]
        S_r = S[:r]
        Vt_r = Vt[:r, :]
        L = (U_r * S_r.unsqueeze(0)) @ Vt_r
        residual = W - L

        if self.structured_24:
            residual_sp = vectorized_24_prune(residual)
        else:
            residual_sp = unstructured_prune(residual, self.sparsity)

        W_recon = L + residual_sp
        err = float(
            torch.linalg.norm(W - W_recon, ord="fro").item() / max(norm_w, 1e-12)
        )
        return W_recon, err

    # ------------------------------------------------------------------
    # Compress
    # ------------------------------------------------------------------
    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_ranks = {}
        self.layer_sparsities = {}
        self.layer_errors = {}

        total_orig_params = 0
        total_comp_params = 0
        effective_sparse_frac = 0.5 if self.structured_24 else (1.0 - self.sparsity)

        # EN-FIX-05: single source of truth
        target_list = [
            (name, module)
            for name, module in self.model.named_modules()
            if self._is_target_module(name, module)
        ]

        if not target_list:
            # EN-FIX-03: diagnostic breakdown
            logger.warning("ENLighten: no compressible layers found. Running diagnostic...")
            total_scanned = 0
            for name, module in self.model.named_modules():
                total_scanned += 1
                self._is_target_module(name, module, _debug=True)
            logger.warning("ENLighten diagnostic: %d modules scanned, 0 targets.", total_scanned)
            return self.model, self.get_compression_stats()

        logger.info("ENLighten: %d target layers identified.", len(target_list))

        for name, module in target_list:
            target_mod = getattr(module, "base_layer", module)
            cls_name = target_mod.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
            raw_w = target_mod.weight.data

            if is_conv1d:
                in_f, out_f = raw_w.shape[0], raw_w.shape[1]
                W = raw_w.t().contiguous().float()
            else:
                out_f, in_f = raw_w.shape[0], raw_w.shape[1]
                W = raw_w.float()

            orig_layer_params = out_f * in_f
            total_orig_params += orig_layer_params

            max_r = max(1, min(out_f, in_f))
            current_r = max(1, min(self.rank, max_r))
            best_w_new: Optional[torch.Tensor] = None
            best_r = current_r
            final_err = 1.0

            for attempt in range(self.max_retries + 1):
                try:
                    w_cand, err = self._decompose_and_prune(W, current_r)
                    final_err = err
                    best_w_new = w_cand
                    best_r = current_r
                    if err <= self.max_error:
                        break
                except Exception as e:
                    logger.debug("ENLighten decomposition failed on %s (iter %d): %s",
                                 name, attempt, e)
                    break

                if attempt < self.max_retries:
                    next_r = min(int(current_r * 1.5), max_r)
                    if next_r <= current_r:
                        break
                    logger.debug(
                        "ENLighten: Error %.4f > %.4f on %s. Escalating rank %d → %d.",
                        err, self.max_error, name, current_r, next_r,
                    )
                    current_r = next_r

            self.layer_ranks[name] = best_r  # EN-FIX-06: actual rank used
            self.layer_sparsities[name] = 0.5 if self.structured_24 else self.sparsity
            self.layer_errors[name] = final_err

            if best_w_new is None or final_err > self.max_error:
                logger.warning(
                    "ENLighten: error %.4f on %s > %.4f. %s",
                    final_err, name, self.max_error,
                    "Falling back to original." if self.fallback_to_original else "Keeping compressed.",
                )
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    total_comp_params += orig_layer_params
                    continue

            # Write back in original orientation and dtype
            if is_conv1d:
                target_mod.weight.data.copy_(
                    best_w_new.t().contiguous().to(
                        dtype=target_mod.weight.dtype, device=target_mod.weight.device)
                )
            else:
                target_mod.weight.data.copy_(
                    best_w_new.to(
                        dtype=target_mod.weight.dtype, device=target_mod.weight.device)
                )

            # Preserve LoRA trainability
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                for attr in ("lora_A", "lora_B"):
                    sub = getattr(module, attr, None)
                    if sub is None:
                        continue
                    if hasattr(sub, "weight") and sub.weight is not None:
                        sub.weight.requires_grad = True

            self.compressed_layers.append(name)
            # EN-FIX-07: honest parameter accounting.
            # Low-rank storage + 50% (or 1-sparsity) residual density.
            comp_layer_params = best_r * (in_f + out_f) + int(effective_sparse_frac * orig_layer_params)
            total_comp_params += comp_layer_params

        # For framework compatibility, count actual params (may equal before).
        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        effective_ratio = (
            total_comp_params / total_orig_params if total_orig_params > 0 else 1.0
        )
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers)
            / max(1, len(self.compressed_layers))
            if self.compressed_layers else 0.0
        )

        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        stats.update({
            "rank": self.rank,
            "sparsity": 0.5 if self.structured_24 else self.sparsity,
            "structured_24": self.structured_24,
            "max_error": self.max_error,
            "target_attention_only": self.target_attention_only,
            "fallback_to_original": self.fallback_to_original,
            "compressed_layers": self.compressed_layers,
            "compressed_layer_count": len(self.compressed_layers),
            "fallback_layers": self.fallback_layers,
            "fallback_layer_count": len(self.fallback_layers),
            "skipped_layers": self.skipped_layers,
            "skipped_layer_count": len(self.skipped_layers),
            "layer_ranks": self.layer_ranks,
            "layer_sparsities": self.layer_sparsities,
            "layer_errors": self.layer_errors,
            "avg_reconstruction_error": avg_err,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "effective_parameter_ratio": round(effective_ratio, 4),
            # EN-FIX-07: metric of true memory saving (excluding sparse index overhead).
            "sparse_density": effective_sparse_frac,
            "method": "enlighten",
        })
        self.stats = stats
        logger.info(
            "ENLighten: %d compressed, %d fallback. Avg error %.4f, Effective Param Ratio %.2fx.",
            len(self.compressed_layers), len(self.fallback_layers), avg_err, effective_ratio,
        )
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================

@COMPRESSOR_REGISTRY.register("enlighten")
@COMPRESSOR_REGISTRY.register("enlighten_compressor")
@COMPRESSOR_REGISTRY.register("enlighten_factory")
def enlighten_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """ENLighten compressor factory."""
    rank = kwargs.pop("rank", _cfg_get(comp_config, "enlighten_rank",
                                       _cfg_get(comp_config, "rank", 64)))
    sparsity = kwargs.pop("sparsity", _cfg_get(comp_config, "enlighten_sparsity",
                                               _cfg_get(comp_config, "sparsity", 0.5)))
    structured_24 = kwargs.pop("structured_24",
                               _cfg_get(comp_config, "enlighten_structured_24",
                                        _cfg_get(comp_config, "structured_24", True)))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "enlighten_max_error",
                                                 _cfg_get(comp_config, "max_error", 0.35)))
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "enlighten_target_attention_only",
                 _cfg_get(comp_config, "target_attention_only", False)),
    )
    fallback = kwargs.pop("fallback_to_original",
                          _cfg_get(comp_config, "enlighten_fallback_to_original",
                                   _cfg_get(comp_config, "fallback_to_original", True)))
    skip_filter = kwargs.pop("skip_should_compress_filter",
                             _cfg_get(comp_config, "skip_should_compress_filter", False))

    compressor = ENLightenCompressor(
        model,
        rank=rank,
        sparsity=sparsity,
        structured_24=structured_24,
        max_error=max_error,
        target_attention_only=target_attention_only,
        fallback_to_original=fallback,
        skip_should_compress_filter=skip_filter,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = [
    "ENLightenCompressor",
    "enlighten_factory",
    "vectorized_24_prune",
    "unstructured_prune",
]