"""
LSC: Low-Rank Spectral & Fourier-Domain Linear Compressor.
Registered in COMPRESSOR_REGISTRY as 'lsc' and 'lsc_compressor'.

Mathematical Formulation:
1. 2D Discrete Fourier Transform:
   $$\\widehat{W} = \\mathcal{F}_{2D}(W) \\in \\mathbb{C}^{d_{out} \\times d_{in}}$$
2. Spectral Magnitude Spectrum:
   $$M_{ij} = |\\widehat{W}_{ij}| = \\sqrt{\\operatorname{Re}(\\widehat{W}_{ij})^2 + \\operatorname{Im}(\\widehat{W}_{ij})^2}$$
3. Spectral Coefficient Selection:
   $$\\mathcal{M}_{spectral} = \\operatorname{Top-}k(M), \\quad k = \\min(r \\cdot r, d_{out} \\cdot d_{in})$$
4. Inverse 2D Fourier Reconstruction:
   $$W_{lsc} = \\operatorname{Re}\\left( \\mathcal{F}_{2D}^{-1}(\\widehat{W} \\odot \\mathcal{M}_{spectral}) \\right)$$
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


class LSCPrunedLoRALinear(nn.Module):
    """
    Linear layer containing 2D Fourier spectral filtered base weights (frozen)
    with optional trainable LoRA adapters (lora_A, lora_B).
    """

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
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, in_features), dtype=torch.float32, device=target_device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_r), dtype=torch.float32, device=target_device)
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
    ) -> LSCPrunedLoRALinear:
        base_layer = getattr(linear, "base_layer", linear)
        bias = base_layer.bias.detach() if hasattr(base_layer, "bias") and base_layer.bias is not None else None
        in_features = getattr(base_layer, "in_features", weight.shape[1])
        out_features = getattr(base_layer, "out_features", weight.shape[0])

        existing_lora_r = getattr(linear, "r", getattr(linear, "lora_r", 0))
        eff_lora_r = existing_lora_r if existing_lora_r > 0 else lora_r
        eff_lora_alpha = getattr(linear, "lora_alpha", getattr(linear, "alpha", lora_alpha))

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

        # Preserve existing LoRA weights if present
        if hasattr(linear, "lora_A") and hasattr(linear, "lora_B") and layer.lora_A is not None and layer.lora_B is not None:
            with torch.no_grad():
                src_A = linear.lora_A.weight.data if hasattr(linear.lora_A, "weight") else linear.lora_A.data
                src_B = linear.lora_B.weight.data if hasattr(linear.lora_B, "weight") else linear.lora_B.data
                if src_A.shape == layer.lora_A.shape and src_B.shape == layer.lora_B.shape:
                    layer.lora_A.copy_(src_A.to(layer.lora_A.device, dtype=layer.lora_A.dtype))
                    layer.lora_B.copy_(src_B.to(layer.lora_B.device, dtype=layer.lora_B.dtype))

        if layer.lora_A is not None:
            layer.lora_A.requires_grad = True
        if layer.lora_B is not None:
            layer.lora_B.requires_grad = True

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + self.scaling * F.linear(
                F.linear(x.float(), self.lora_A), self.lora_B
            ).to(out.dtype)
        return out


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


@COMPRESSOR_REGISTRY.register("lsc")
@COMPRESSOR_REGISTRY.register("lsc_compressor")
class LSCCompressor(BaseCompressor):
    """
    LSC Compressor: 2D Fourier spectral low-rank and frequency filtering compressor.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error (default 0.85).
    - Safe fallback to original weights by default (fallback_to_original=True).
    - Robust Conv1D detection and support across GPT-2 architectures.
    - Unified critical-layer protection via should_compress_linear.
    - GQA-aware skipping to preserve key-value cache dimension alignment.
    - Tied / shared weight protection tracking parameter IDs.
    - Trainable LoRA adapter integration with base weights frozen.
    - FP32 numerical stability with NaN/Inf guards.
    - Post-compression forward sanity check under torch.no_grad().
    - Comprehensive per-layer stats and error tracking.
    """

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
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="lsc", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = _cfg_get(cfg, "lsc_rank", _cfg_get(cfg, "rank", rank))
            energy_ratio = _cfg_get(cfg, "lsc_energy_ratio", _cfg_get(cfg, "energy_ratio", energy_ratio))
            max_error = _cfg_get(cfg, "lsc_max_error", _cfg_get(cfg, "max_error", max_error))
            fallback_to_original = _cfg_get(
                cfg, "lsc_fallback_to_original", _cfg_get(cfg, "fallback_to_original", fallback_to_original)
            )
            lora_r = _cfg_get(cfg, "lora_r", lora_r)
            lora_alpha = _cfg_get(cfg, "lora_alpha", lora_alpha)
            skip_attention_qkv = _cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv)
            target_attention_only = _cfg_get(cfg, "target_attention_only", target_attention_only)

        self.rank = max(1, int(rank))
        self.energy_ratio = float(energy_ratio) if energy_ratio is not None else None
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.target_attention_only = bool(target_attention_only)
        self.target_modules = target_modules

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
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

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes 2D Fourier spectral compression across all Linear and Conv1D layers."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        layer_details: Dict[str, Any] = {}
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

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                _add_params(module)
                continue

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

            # FP32 computation
            W = raw_w.t().clone().float() if is_conv1d else raw_w.clone().float()
            d_out, d_in = W.shape
            total_coeffs = d_out * d_in

            try:
                # 2D FFT in FP32
                W_fft = torch.fft.fft2(W)
                magnitude = torch.abs(W_fft)

                if self.energy_ratio is not None:
                    flat_mag = magnitude.reshape(-1)
                    sorted_mag, _ = torch.sort(flat_mag, descending=True)
                    cum_energy = torch.cumsum(sorted_mag ** 2, dim=0) / torch.sum(sorted_mag ** 2).clamp(min=1e-8)
                    idx = torch.where(cum_energy >= self.energy_ratio)[0]
                    k = idx[0].item() + 1 if len(idx) > 0 else total_coeffs
                    k = max(1, min(k, total_coeffs))
                else:
                    k = max(1, min(self.rank * self.rank, total_coeffs))

                # Top-k spectral mask
                flat_mag = magnitude.reshape(-1)
                thr = torch.topk(flat_mag, k=k, largest=True).values[-1]
                spectral_mask = (magnitude >= thr).float()

                # Inverse 2D FFT
                W_comp = torch.fft.ifft2(W_fft * spectral_mask).real
                W_comp = torch.nan_to_num(W_comp, nan=0.0)

                # Reconstruction error check
                norm_orig = torch.norm(W, p="fro").item()
                rel_err = (torch.norm(W - W_comp, p="fro") / max(norm_orig, 1e-12)).item()
                self.layer_errors[name] = float(rel_err)

                if not math.isfinite(rel_err) or rel_err > self.max_error:
                    logger.warning(
                        "LSC reconstruction error %.4f on %s exceeded max_error %.4f%s",
                        rel_err,
                        name,
                        self.max_error,
                        ". Keeping original layer." if self.fallback_to_original else ". Proceeding with filtered weights.",
                    )
                    if self.fallback_to_original:
                        self.fallback_layers.append(name)
                        for p in module.parameters(recurse=False):
                            p.requires_grad = False
                        _add_params(module)
                        continue

                new_w = W_comp.t() if is_conv1d else W_comp
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
                    "k_coeffs": k,
                    "rel_error": rel_err,
                    "shape": tuple(W_comp.shape),
                }
            except Exception as e:
                logger.warning("LSC compression exception on %s: %s. Keeping original layer.", name, e)
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
            logger.warning("LSC post-compression forward check warning: %s", e)
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers) / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        eff_ratio = total_comp_params / max(1, total_orig_params)
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
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("lsc_factory")
def lsc_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for LSC compressor."""
    rank = _cfg_get(comp_config, "lsc_rank", _cfg_get(comp_config, "rank", kwargs.get("rank", 32)))
    energy_ratio = _cfg_get(comp_config, "lsc_energy_ratio", _cfg_get(comp_config, "energy_ratio", kwargs.get("energy_ratio", None)))
    max_error = _cfg_get(comp_config, "max_error", kwargs.get("max_error", 0.85))
    fallback_to_original = _cfg_get(comp_config, "fallback_to_original", kwargs.get("fallback_to_original", True))
    lora_r = _cfg_get(comp_config, "lora_r", kwargs.pop("lora_r", 0))
    lora_alpha = _cfg_get(comp_config, "lora_alpha", kwargs.pop("lora_alpha", 1.0))
    target_attention_only = _cfg_get(comp_config, "target_attention_only", kwargs.pop("target_attention_only", False))
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
