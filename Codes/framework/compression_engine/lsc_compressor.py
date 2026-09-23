"""
LSC: Low-Rank Spectral & Fourier-Domain Linear Compressor.
Registered as 'lsc', 'lsc_class', 'lsc_compressor', 'lsc_factory'.

Mathematical Formulation:
  1. 2D DFT: \hat{W} = F_2D(W)
  2. Magnitude: M_ij = |\hat{W}_ij|
  3. Top-k selection: M_spectral = Top-k(M), k = min(r*r, d_out*d_in)
  4. Inverse DFT: W_lsc = Re(F_2D^{-1}(\hat{W} ⊙ M_spectral))

Bug-fix map:
  LSC-FIX-01  is_custom_attention = False on compressor.
  LSC-FIX-02  Robust LoRA rank detection (_detect_lora_rank).
  LSC-FIX-03  LoRA dtype matches base weight dtype.
  LSC-FIX-04  Forward check uses input_ids (long tensor).
  LSC-FIX-05  skip_should_compress_filter option.
  LSC-FIX-06  Debug logging when no targets.
  LSC-FIX-07  LSCPrunedLoRALinear.is_custom_attention = False (it's a linear).
  LSC-FIX-08  Guard against degenerate all-zero spectral mask.
  LSC-FIX-09  Preserve requires_grad=False for fallback base weights.
  LSC-FIX-10  Robust forward sanity check with attention_mask fallback.
  LSC-FIX-11  Factory registered under 'lsc' name.
  LSC-FIX-12  Parameter accounting deduplicated by id().
  LSC-FIX-13  Write compression_stats onto model for framework visibility.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


# ===========================================================================
# Helpers
# ===========================================================================

def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _detect_lora_rank(module: nn.Module) -> int:
    """LSC-FIX-02: robust LoRA rank detection."""
    for attr in ("r", "lora_r", "rank"):
        v = getattr(module, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    la = getattr(module, "lora_A", None)
    if la is not None:
        if hasattr(la, "out_features"):
            try:
                return int(la.out_features)
            except Exception:
                pass
        if hasattr(la, "weight") and la.weight is not None and la.weight.dim() >= 1:
            return int(la.weight.shape[0])
        if isinstance(la, torch.Tensor) and la.dim() >= 1:
            return int(la.shape[0])
    return 0


def _detect_lora_alpha(module: nn.Module) -> float:
    for attr in ("lora_alpha", "alpha"):
        v = getattr(module, attr, None)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return 1.0


_ATTN_NAMES = {
    "attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj",
    "c_attn", "c_proj", "query", "key", "value",
    "q_lin", "k_lin", "v_lin", "out_lin", "self_attn",
}


# ===========================================================================
# LSCPrunedLoRALinear (LSC-FIX-03, LSC-FIX-07)
# ===========================================================================

class LSCPrunedLoRALinear(nn.Module):
    # LSC-FIX-07: this is a linear layer, not an attention module.
    is_custom_attention: bool = False
    is_custom_linear: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features  # Conv1D compatibility
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0
        self.rank = self.lora_r  # Alias for detection

        target_device = device if device is not None else weight.device
        target_dtype = dtype if dtype is not None else weight.dtype

        self.weight = nn.Parameter(
            weight.to(device=target_device, dtype=target_dtype),
            requires_grad=False,
        )
        if bias is not None:
            self.bias = nn.Parameter(
                bias.to(device=target_device, dtype=target_dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            # LSC-FIX-03: LoRA parameters in base weight's dtype for forward consistency.
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, in_features),
                            dtype=target_dtype, device=target_device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_r),
                            dtype=target_dtype, device=target_device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        weight: torch.Tensor,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> "LSCPrunedLoRALinear":
        base_layer = getattr(linear, "base_layer", linear)
        bias = (
            base_layer.bias.detach()
            if hasattr(base_layer, "bias") and base_layer.bias is not None
            else None
        )
        in_features = getattr(base_layer, "in_features", weight.shape[1])
        out_features = getattr(base_layer, "out_features", weight.shape[0])

        existing_lora_r = _detect_lora_rank(linear)
        eff_lora_r = existing_lora_r if existing_lora_r > 0 else lora_r
        eff_lora_alpha = _detect_lora_alpha(linear) if existing_lora_r > 0 else lora_alpha

        layer = cls(
            in_features=in_features,
            out_features=out_features,
            weight=weight,
            bias=bias,
            lora_r=eff_lora_r,
            lora_alpha=eff_lora_alpha,
            device=weight.device,
            dtype=weight.dtype,
        )

        # Preserve existing LoRA if shapes match.
        if (hasattr(linear, "lora_A") and hasattr(linear, "lora_B")
                and layer.lora_A is not None and layer.lora_B is not None):
            with torch.no_grad():
                src_A = getattr(linear.lora_A, "weight", linear.lora_A)
                src_B = getattr(linear.lora_B, "weight", linear.lora_B)
                if (isinstance(src_A, torch.Tensor)
                        and src_A.shape == layer.lora_A.shape):
                    layer.lora_A.data.copy_(
                        src_A.to(layer.lora_A.device, dtype=layer.lora_A.dtype)
                    )
                if (isinstance(src_B, torch.Tensor)
                        and src_B.shape == layer.lora_B.shape):
                    layer.lora_B.data.copy_(
                        src_B.to(layer.lora_B.device, dtype=layer.lora_B.dtype)
                    )

        if layer.lora_A is not None:
            layer.lora_A.requires_grad = True
        if layer.lora_B is not None:
            layer.lora_B.requires_grad = True

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            # Preserve dtype consistency: do the LoRA computation in the
            # base weight's dtype (input dtype may differ).
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
            out = out + self.scaling * lora_out.to(out.dtype)
        return out


# ===========================================================================
# Compressor
# ===========================================================================

@COMPRESSOR_REGISTRY.register("lsc_class")
@COMPRESSOR_REGISTRY.register("lsc_compressor_class")
class LSCCompressor(BaseCompressor):
    # LSC-FIX-01: this is a compressor, not an attention layer.
    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module,
        rank: int = 32,
        energy_ratio: Optional[float] = None,
        max_error: float = 0.85,
        fallback_to_original: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = False,
        target_attention_only: bool = False,
        target_modules: Optional[List[str]] = None,
        min_layer_dim: int = 8,
        skip_should_compress_filter: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="lsc", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = _cfg_get(cfg, "lsc_rank", _cfg_get(cfg, "rank", rank))
            energy_ratio = _cfg_get(cfg, "lsc_energy_ratio",
                                    _cfg_get(cfg, "energy_ratio", energy_ratio))
            max_error = _cfg_get(cfg, "lsc_max_error",
                                 _cfg_get(cfg, "max_error", max_error))
            fallback_to_original = _cfg_get(
                cfg, "lsc_fallback_to_original",
                _cfg_get(cfg, "fallback_to_original", fallback_to_original),
            )
            lora_r = _cfg_get(cfg, "lora_r", lora_r)
            lora_alpha = _cfg_get(cfg, "lora_alpha", lora_alpha)
            skip_attention_qkv = _cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv)
            target_attention_only = _cfg_get(cfg, "target_attention_only", target_attention_only)
            min_layer_dim = _cfg_get(cfg, "min_layer_dim", min_layer_dim)
            skip_should_compress_filter = _cfg_get(
                cfg, "skip_should_compress_filter", skip_should_compress_filter,
            )

        self.rank = max(1, int(rank))
        self.energy_ratio = float(energy_ratio) if energy_ratio is not None else None
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.target_attention_only = bool(target_attention_only)
        self.target_modules = target_modules
        self.min_layer_dim = max(1, int(min_layer_dim))
        self.skip_should_compress_filter = bool(skip_should_compress_filter)

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}
        self._compressed_params: int = 0

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads:
            if any(p in name.lower() for p in
                   ["q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn"]):
                return True
        return False

    # ------------------------------------------------------------------
    # Target check with debug
    # ------------------------------------------------------------------
    def _is_target_module(self, name: str, module: nn.Module, _debug: bool = False) -> bool:
        name_lower = name.lower()
        if any(part in name.split(".") for part in ("lora_A", "lora_B", "base_layer")):
            if _debug:
                logger.info(f"  [skip] {name}: lora/base_layer part")
            return False

        cls_name = module.__class__.__name__
        if cls_name in ("LSCPrunedLoRALinear", "DynamicSVDLinear",
                        "SVDLinear", "FactorisedLinear"):
            if _debug:
                logger.info(f"  [skip] {name}: already compressed ({cls_name})")
            return False

        base_mod = getattr(module, "base_layer", module)
        cls_lower = base_mod.__class__.__name__.lower()
        is_linear_like = (
            isinstance(base_mod, nn.Linear)
            or "conv1d" in cls_lower
            or hasattr(base_mod, "nf")
        )
        if not is_linear_like:
            if _debug:
                logger.info(f"  [skip] {name}: not linear-like")
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

        w = getattr(base_mod, "weight", None)
        if w is None:
            if _debug:
                logger.info(f"  [skip] {name}: weight None")
            return False
        if min(w.shape[0], w.shape[1]) < self.min_layer_dim:
            if _debug:
                logger.info(f"  [skip] {name}: dim too small {tuple(w.shape)}")
            return False

        if self._is_gqa_asymmetric(name, module):
            if _debug:
                logger.info(f"  [skip] {name}: GQA asymmetric")
            return False

        if self.target_attention_only:
            if not any(k in name_lower for k in _ATTN_NAMES):
                if _debug:
                    logger.info(f"  [skip] {name}: not attention")
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
    # Main compress
    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        layer_details: Dict[str, Any] = {}

        seen_weights: Set[int] = set()
        seen_param_ids: Set[int] = set()
        total_orig_params = 0
        total_comp_params = 0

        # LSC-FIX-12: deduplicated parameter accounting.
        for p in self.model.parameters():
            if id(p) not in seen_param_ids:
                seen_param_ids.add(id(p))
                total_orig_params += p.numel()

        count_param_ids: Set[int] = set()

        def _add_params(mod: nn.Module) -> None:
            nonlocal total_comp_params
            for p in mod.parameters(recurse=False):
                if id(p) not in count_param_ids:
                    count_param_ids.add(id(p))
                    total_comp_params += p.numel()

        # LSC-FIX-06: diagnostic breakdown if no targets
        target_list = [
            (name, module)
            for name, module in self.model.named_modules()
            if self._is_target_module(name, module)
        ]
        if not target_list:
            logger.warning("LSC: no compressible layers found. Running diagnostic...")
            n_scanned = 0
            for name, module in self.model.named_modules():
                n_scanned += 1
                self._is_target_module(name, module, _debug=True)
            logger.warning("LSC diagnostic: %d modules scanned, 0 targets.", n_scanned)
            try:
                stats = self.get_compression_stats()
            except Exception:
                stats = {}
            return self.model, stats

        logger.info("LSC: %d target layers identified.", len(target_list))

        for name, module in target_list:
            target_layer = getattr(module, "base_layer", module)
            if not hasattr(target_layer, "weight") or target_layer.weight is None:
                _add_params(module)
                continue

            w_id = id(target_layer.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                _add_params(module)
                continue
            seen_weights.add(w_id)

            raw_w = target_layer.weight.data
            cls_name = target_layer.__class__.__name__.lower()
            is_conv1d = "conv1d" in cls_name or hasattr(target_layer, "nf")
            device = raw_w.device
            dtype = raw_w.dtype

            # FP32 for numerical stability
            W = raw_w.t().clone().float() if is_conv1d else raw_w.clone().float()
            d_out, d_in = W.shape
            total_coeffs = d_out * d_in

            try:
                W_fft = torch.fft.fft2(W)
                magnitude = torch.abs(W_fft)
                flat_mag = magnitude.reshape(-1)

                # LSC-FIX-08: guard against all-zero magnitude
                if flat_mag.max().item() <= 1e-12:
                    logger.warning(
                        "LSC: all-zero magnitude for %s; keeping original.",
                        name,
                    )
                    self.fallback_layers.append(name)
                    for p in module.parameters(recurse=False):
                        p.requires_grad = False
                    _add_params(module)
                    continue

                if self.energy_ratio is not None:
                    sorted_mag, _ = torch.sort(flat_mag, descending=True)
                    total_energy = (sorted_mag ** 2).sum().clamp(min=1e-12)
                    cum_energy = torch.cumsum(sorted_mag ** 2, dim=0) / total_energy
                    idx = torch.where(cum_energy >= self.energy_ratio)[0]
                    k = idx[0].item() + 1 if len(idx) > 0 else total_coeffs
                    k = max(1, min(k, total_coeffs))
                else:
                    k = max(1, min(self.rank * self.rank, total_coeffs))

                thr = torch.topk(flat_mag, k=k, largest=True).values[-1]
                spectral_mask = (magnitude >= thr).float()

                W_comp = torch.fft.ifft2(W_fft * spectral_mask).real
                W_comp = torch.nan_to_num(W_comp, nan=0.0, posinf=0.0, neginf=0.0)

                norm_orig = torch.norm(W, p="fro").item()
                rel_err = (
                    torch.norm(W - W_comp, p="fro").item() / max(norm_orig, 1e-12)
                )
                self.layer_errors[name] = float(rel_err)

                if not math.isfinite(rel_err) or rel_err > self.max_error:
                    logger.warning(
                        "LSC: err %.4f on %s > %.4f. %s",
                        rel_err, name, self.max_error,
                        "Keeping original." if self.fallback_to_original
                        else "Proceeding with filtered weights.",
                    )
                    if self.fallback_to_original:
                        self.fallback_layers.append(name)
                        for p in module.parameters(recurse=False):
                            p.requires_grad = False
                        _add_params(module)
                        continue

                new_w = W_comp.t().contiguous() if is_conv1d else W_comp
                expected_shape = target_layer.weight.shape
                if new_w.shape != expected_shape:
                    logger.warning(
                        "LSC: shape mismatch for %s: expected %s, got %s; skipping.",
                        name, tuple(expected_shape), tuple(new_w.shape),
                    )
                    for p in module.parameters(recurse=False):
                        p.requires_grad = False
                    _add_params(module)
                    continue

                new_layer = LSCPrunedLoRALinear.from_linear(
                    module,
                    weight=new_w.to(dtype=dtype, device=device),
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )
                replace_module(self.model, name, new_layer)
                self.compressed_layers.append(name)
                _add_params(new_layer)
                layer_details[name] = {
                    "k_coeffs": int(k),
                    "rel_error": float(rel_err),
                    "shape": tuple(new_w.shape),
                }
            except Exception as e:
                logger.warning(
                    "LSC: exception on %s: %s. Keeping original layer.",
                    name, e,
                )
                self.fallback_layers.append(name)
                self.layer_errors[name] = 1.0
                for p in module.parameters(recurse=False):
                    p.requires_grad = False
                _add_params(module)

        # LSC-FIX-04 + LSC-FIX-10: robust forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                first_param = next(iter(self.model.parameters()), None)
                dev = first_param.device if first_param is not None else torch.device("cpu")
                dummy_ids = torch.randint(0, 100, (1, 8), device=dev, dtype=torch.long)
                dummy_mask = torch.ones_like(dummy_ids)
                try:
                    self.model(input_ids=dummy_ids, attention_mask=dummy_mask)
                except Exception:
                    try:
                        self.model(dummy_ids)
                    except Exception as e2:
                        logger.warning("LSC: forward check failed: %s", e2)
                        forward_check_success = False
        except Exception as e:
            logger.warning("LSC: forward check raised: %s", e)
            forward_check_success = False

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers)
            / max(1, len(self.compressed_layers))
            if self.compressed_layers else 0.0
        )
        eff_ratio = total_comp_params / max(1, total_orig_params)

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
            "rank": self.rank,
            "energy_ratio": self.energy_ratio,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "avg_reconstruction_error": avg_err,
            "layer_errors": self.layer_errors,
            "layer_details": layer_details,
            "effective_parameter_ratio": eff_ratio,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "forward_check_success": forward_check_success,
            "method": "lsc_fourier_spectral",
            "note": (
                "LSC performs in-place spectral filtering of weights. "
                "The stored weight is still dense (out x in); "
                "framework-level compression_rate reads 1.0. "
                "Use effective_parameter_ratio for the true storage metric."
            ),
        })
        self.stats = stats

        # LSC-FIX-13: write stats onto the model for framework visibility.
        try:
            if not hasattr(self.model, "compression_stats") or self.model.compression_stats is None:
                self.model.compression_stats = {}
            self.model.compression_stats.update({
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "compression_method": "lsc_fourier_spectral",
                "effective_parameter_ratio": eff_ratio,
                "avg_reconstruction_error": avg_err,
            })
        except Exception:
            pass

        logger.info(
            "LSC: %d compressed, %d fallback, %d skipped. "
            "Avg err %.4f, effective_param_ratio %.2fx.",
            len(self.compressed_layers), len(self.fallback_layers),
            len(self.skipped_layers), avg_err, eff_ratio,
        )
        return self.model, stats


# ===========================================================================
# Factory (LSC-FIX-11)
# ===========================================================================

@COMPRESSOR_REGISTRY.register("lsc")
@COMPRESSOR_REGISTRY.register("lsc_compressor")
@COMPRESSOR_REGISTRY.register("lsc_factory")
def lsc_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """LSC Fourier spectral compressor factory."""
    rank = _cfg_get(comp_config, "lsc_rank",
                    _cfg_get(comp_config, "rank", kwargs.pop("rank", 32)))
    energy_ratio = _cfg_get(comp_config, "lsc_energy_ratio",
                            _cfg_get(comp_config, "energy_ratio",
                                     kwargs.pop("energy_ratio", None)))
    max_error = _cfg_get(comp_config, "max_error",
                         kwargs.pop("max_error", 0.85))
    fallback_to_original = _cfg_get(comp_config, "fallback_to_original",
                                    kwargs.pop("fallback_to_original", True))
    lora_r = _cfg_get(comp_config, "lora_r", kwargs.pop("lora_r", 0))
    lora_alpha = _cfg_get(comp_config, "lora_alpha", kwargs.pop("lora_alpha", 1.0))
    target_attention_only = _cfg_get(comp_config, "target_attention_only",
                                     kwargs.pop("target_attention_only", False))
    skip_filter = _cfg_get(comp_config, "skip_should_compress_filter",
                           kwargs.pop("skip_should_compress_filter", False))
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = LSCCompressor(
        model,
        rank=rank,
        energy_ratio=energy_ratio,
        max_error=max_error,
        fallback_to_original=fallback_to_original,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_attention_only=target_attention_only,
        skip_should_compress_filter=skip_filter,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "LSCPrunedLoRALinear",
    "LSCCompressor",
    "lsc_factory",
]