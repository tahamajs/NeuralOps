"""
ERC-SVD: Error-Controlled SVD with Bounded Frobenius Truncation Error.
Registered as 'erc' and 'erc_svd'.

Bug-fix map:
  ER-FIX-01  fallback path handles svd_layer=None safely.
  ER-FIX-02  LoRA rank detected via lora_A.weight.shape[0] (PEFT-compatible).
  ER-FIX-03  forward sanity check error message corrected.
  ER-FIX-04  attention_mask included in forward check.
  ER-FIX-05  r_init clamped to max_r before storing.
  ER-FIX-06  erc_factory guarded, returns stats optionally.
  ER-FIX-07  no wasted SVDLayer when best_err > max_error and fallback enabled.
  ER-FIX-08  no silent mutation of non-target layers (only freeze when needed).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

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
    "q_proj", "k_proj", "v_proj", "o_proj",
    "c_attn", "c_proj", "query", "key", "value",
    "q_lin", "k_lin", "v_lin", "out_lin", "out_proj", "dense",
}


@COMPRESSOR_REGISTRY.register("erc")
@COMPRESSOR_REGISTRY.register("erc_svd")
class ERCCompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    Error-Controlled SVD (ERC-SVD) Compressor.

    Finds the minimal rank r per layer such that
        ||W - W_r||_F / ||W||_F <= allowed_error.
    Factored symmetrically (U_r sqrt(S), sqrt(S) V_r^T) and replaces with SVDLinear.
    """

    def __init__(
        self,
        model: nn.Module,
        allowed_error: float = 0.20,
        max_error: Optional[float] = None,
        fallback_to_original: bool = True,
        target_attention_only: bool = False,
        rank: Optional[int] = None,
        max_retries: int = 2,
        target_modules: Optional[List[str]] = None,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="erc", **kwargs)
        cfg = cfg or kwargs.get("comp_config", None)
        self.cfg = cfg

        self.allowed_error = float(
            _cfg_get(cfg, "erc_allowed_error", _cfg_get(cfg, "allowed_error", allowed_error))
        )
        effective_max = _cfg_get(cfg, "erc_max_error",
                                _cfg_get(cfg, "max_error", max_error))
        self.max_error = float(effective_max if effective_max is not None else self.allowed_error)
        self.target_attention_only = bool(
            _cfg_get(cfg, "erc_target_attention_only",
                     _cfg_get(cfg, "target_attention_only", target_attention_only))
        )
        self.fallback_to_original = bool(
            _cfg_get(cfg, "erc_fallback_to_original",
                     _cfg_get(cfg, "fallback_to_original", fallback_to_original))
        )
        self.rank = _cfg_get(cfg, "erc_rank", _cfg_get(cfg, "rank", rank))
        if self.rank is not None:
            self.rank = int(self.rank)
        self.max_retries = max(0, int(max_retries))
        self.target_modules = target_modules

        # Pre-LayerNorm encoder detection (e.g., RoBERTa): pre-LN amplifies residual truncation noise.
        # Enforce stricter maximum error threshold (0.15) to prevent below-chance catastrophic collapse.
        model_name_or_cls = (
            getattr(getattr(model, "config", None), "_name_or_path", "")
            or model.__class__.__name__
        ).lower()
        self.is_pre_layernorm_encoder = "roberta" in model_name_or_cls
        if self.is_pre_layernorm_encoder and self.max_error > 0.15:
            logger.info(
                "ERC-SVD: Detected pre-LayerNorm encoder (%s). Capping max_error from %.2f to 0.15.",
                model_name_or_cls, self.max_error,
            )
            self.max_error = min(self.max_error, 0.15)

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_ranks: Dict[str, int] = {}
        self.layer_errors: Dict[str, float] = {}
        self._compressed_params: int = 0

    # ------------------------------------------------------------------
    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
            return False
        if any(skip in name_lower for skip in
               ["classifier", "score", "lm_head", "pooler", "embedding",
                "router", "gate", "head", "output_layer"]):
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_linear = isinstance(target_mod, nn.Linear)
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        if not (is_linear or is_conv1d):
            return False
        if isinstance(target_mod, SVDLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)

        if self.target_attention_only:
            last_part = name.split(".")[-1]
            is_attn = (last_part in _ATTN_NAMES) or any(k in name_lower for k in ("attn", "attention"))
            if not is_attn:
                return False

        return True

    # ------------------------------------------------------------------
    # ER-FIX-02: PEFT-compatible LoRA rank detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_lora(module: nn.Module) -> Tuple[int, float]:
        """Return (lora_rank, lora_alpha) from an existing LoRA wrapper if present."""
        if not hasattr(module, "lora_A"):
            return 0, 1.0
        rank = 0
        la = getattr(module, "lora_A", None)
        if la is not None:
            if hasattr(la, "weight") and la.weight is not None and la.weight.dim() >= 1:
                rank = int(la.weight.shape[0])
            elif isinstance(la, nn.Parameter) and la.dim() >= 1:
                rank = int(la.shape[0])
        alpha = 1.0
        for attr in ("lora_alpha", "alpha"):
            a = getattr(module, attr, None)
            if a is not None:
                try:
                    alpha = float(a)
                except (TypeError, ValueError):
                    pass
                break
        return rank, alpha

    # ------------------------------------------------------------------
    def compress_linear(
        self,
        module: nn.Module,
    ) -> Tuple[Optional[SVDLinear], float, int]:
        """Decompose a single linear, Conv1D, or LoRALinear layer via error-controlled SVD."""
        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        raw_w = target_mod.weight.data

        if is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.t().contiguous().float()
        else:
            out_f, in_f = raw_w.shape
            W = raw_w.float()

        bound_r = (in_f * out_f) // (in_f + out_f)
        max_r = min(in_f, out_f, bound_r)
        if max_r < 16:
            return None, 1.0, 0

        device = raw_w.device
        dtype = raw_w.dtype
        norm_w = torch.norm(W, p="fro").item()

        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        except Exception as e:
            logger.warning("ERC-SVD SVD computation failed: %s", e)
            return None, 1.0, 0

        s_squared = S ** 2
        total_energy = s_squared.sum()
        if total_energy <= 1e-12:
            r_init = 1
        else:
            cum_energy = torch.cumsum(s_squared, dim=0) / total_energy
            target_retention = max(0.0, 1.0 - (self.allowed_error ** 2))
            idx = torch.where(cum_energy >= target_retention)[0]
            r_init = int(idx[0].item()) + 1 if len(idx) > 0 else max_r

        # ER-FIX-05: clamp r_init to max_r before use.
        r_init = min(r_init, max_r)
        if self.rank is not None:
            r_init = min(r_init, int(self.rank))
        current_r = max(1, min(r_init, max_r))

        best_svd_layer: Optional[SVDLinear] = None
        best_err = 1.0
        chosen_r = current_r

        lora_r, lora_alpha = self._detect_lora(module)

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

            svd_layer = SVDLinear(
                in_features=in_f,
                out_features=out_f,
                rank=r,
                bias=(getattr(target_mod, "bias", None) is not None and target_mod.bias is not None),
                dtype=dtype,
                device=device,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
            )
            svd_layer.set_low_rank_factors(U_r, V_r)
            if (getattr(target_mod, "bias", None) is not None
                    and target_mod.bias is not None
                    and svd_layer.bias is not None):
                svd_layer.bias.data.copy_(target_mod.bias.data)

            if (lora_r > 0
                    and getattr(svd_layer, "lora_A", None) is not None
                    and getattr(svd_layer, "lora_B", None) is not None):
                try:
                    src_la = getattr(module, "lora_A", None)
                    src_lb = getattr(module, "lora_B", None)
                    la_w = getattr(src_la, "weight", src_la)
                    lb_w = getattr(src_lb, "weight", src_lb)
                    if hasattr(la_w, "data") and hasattr(svd_layer.lora_A, "data"):
                        svd_layer.lora_A.data.copy_(la_w.data)
                    if hasattr(lb_w, "data") and hasattr(svd_layer.lora_B, "data"):
                        svd_layer.lora_B.data.copy_(lb_w.data)
                except Exception as e:
                    logger.debug("ERC-SVD LoRA copy failed on %s: %s", getattr(module, "_name", "?"), e)

            best_svd_layer = svd_layer

            if err <= self.max_error or r >= max_r:
                break

            if attempt < self.max_retries:
                next_r = min(int(r * 1.5), max_r)
                if next_r <= r:
                    break
                current_r = next_r

        if best_svd_layer is not None:
            best_svd_layer.reconstruction_error = best_err
        return best_svd_layer, best_err, chosen_r

    # ------------------------------------------------------------------
    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress all target linear and Conv1D layers across the model."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_ranks = {}
        self.layer_errors = {}

        total_orig_params = 0
        total_comp_params = 0

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

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("ERC-SVD: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        for name, module in target_layers:
            target_mod = getattr(module, "base_layer", module)
            raw_w = target_mod.weight.data
            orig_params = raw_w.numel()
            total_orig_params += orig_params

            svd_layer, err, r = self.compress_linear(module)
            self.layer_errors[name] = err
            self.layer_ranks[name] = r

            # ER-FIX-01: safe fallback when svd_layer is None.
            if svd_layer is None:
                logger.warning(
                    "ERC-SVD: could not compress %s (rank floor not met or SVD failed).",
                    name,
                )
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                total_comp_params += orig_params
                continue

            if err > self.max_error:
                logger.warning(
                    "ERC-SVD: reconstruction error %.4f on %s exceeded threshold %.4f.",
                    err, name, self.max_error,
                )
                if self.fallback_to_original:
                    logger.info("ERC-SVD: fallback to original on %s.", name)
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    total_comp_params += orig_params
                    continue
                # fallback_to_original=False: proceed with high error (warned).

            replace_module(self.model, name, svd_layer)
            self.compressed_layers.append(name)
            comp_params = r * (svd_layer.in_features + svd_layer.out_features)
            total_comp_params += comp_params

        # ER-FIX-03/04: forward sanity check with attention_mask.
        forward_check_success = True
        try:
            with torch.no_grad():
                first_p = next(self.model.parameters(), None)
                dev = first_p.device if first_p is not None else torch.device("cpu")
                dummy = torch.randint(0, 100, (1, 8), device=dev, dtype=torch.long)
                attn = torch.ones_like(dummy)
                try:
                    self.model(input_ids=dummy, attention_mask=attn)
                except Exception:
                    try:
                        self.model(dummy)
                    except Exception as e_inner:
                        logger.debug("ERC-SVD forward check (input_ids) failed: %s", e_inner)
                        first_linear = next(
                            (m for m in self.model.modules() if isinstance(m, nn.Linear)),
                            None,
                        )
                        in_dim = getattr(first_linear, "in_features", 8)
                        self.model(torch.randn(
                            1, 8, in_dim, device=dev,
                            dtype=first_p.dtype if first_p is not None else torch.float32,
                        ))
        except Exception as e:
            logger.debug("ERC-SVD forward sanity check failed: %s", e)
            forward_check_success = False

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        effective_ratio = (total_comp_params / total_orig_params) if total_orig_params > 0 else 1.0
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers else 0.0
        )
        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}
        stats.update({
            "compressed_layers": self.compressed_layers,
            "compressed_layer_count": len(self.compressed_layers),
            "fallback_layers": self.fallback_layers,
            "fallback_layer_count": len(self.fallback_layers),
            "skipped_layers": self.skipped_layers,
            "skipped_layer_count": len(self.skipped_layers),
            "layer_ranks": self.layer_ranks,
            "layer_errors": self.layer_errors,
            "allowed_error": self.allowed_error,
            "max_error": self.max_error,
            "target_attention_only": self.target_attention_only,
            "fallback_to_original": self.fallback_to_original,
            "avg_reconstruction_error": avg_err,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            "effective_parameter_ratio": round(effective_ratio, 4),
            "forward_check_success": forward_check_success,
            "method": "erc_svd",
        })
        logger.info(
            "ERC-SVD compression complete: %d compressed, %d fallback. "
            "Avg error: %.4f, Effective Param Ratio: %.2fx",
            len(self.compressed_layers), len(self.fallback_layers),
            avg_err, effective_ratio,
        )
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================

@COMPRESSOR_REGISTRY.register("erc_factory")
@COMPRESSOR_REGISTRY.register("erc_svd_factory")
@COMPRESSOR_REGISTRY.register("erc_svd_compressor_factory")
def erc_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """ERC factory function."""
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    allowed_error = kwargs.pop("allowed_error",
                    _cfg_get(comp_config, "erc_allowed_error",
                    _cfg_get(comp_config, "allowed_error", 0.20)))
    max_error = kwargs.pop("max_error",
                _cfg_get(comp_config, "erc_max_error",
                _cfg_get(comp_config, "max_error", allowed_error)))
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "erc_target_attention_only",
                 _cfg_get(comp_config, "target_attention_only", False)),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "erc_fallback_to_original",
                 _cfg_get(comp_config, "fallback_to_original", True)),
    )
    rank = kwargs.pop("rank",
           _cfg_get(comp_config, "erc_rank",
           _cfg_get(comp_config, "rank", None)))
    max_retries = kwargs.pop("max_retries",
                   _cfg_get(comp_config, "max_retries", 2))

    compressor = ERCCompressor(
        model,
        allowed_error=allowed_error,
        max_error=max_error,
        fallback_to_original=fallback,
        target_attention_only=target_attention_only,
        rank=rank,
        max_retries=max_retries,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "ERCCompressor",
    "erc_factory",
]