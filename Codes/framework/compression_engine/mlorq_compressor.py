"""
MLoRQ: Multi-Layer Optimal Rank & Quantisation under a Total Memory Budget.
Registered as 'mlorq' and 'mlorq_compressor'.
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


class MLoRQLoRALinear(nn.Module):
    """
    Linear layer containing MLoRQ low-rank quantized base weights (frozen)
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
        bias: Optional[torch.Tensor] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> MLoRQLoRALinear:
        base_layer = getattr(linear, "base_layer", linear)
        b = bias if bias is not None else (base_layer.bias.detach() if hasattr(base_layer, "bias") and base_layer.bias is not None else None)
        in_features = getattr(base_layer, "in_features", weight.shape[1])
        out_features = getattr(base_layer, "out_features", weight.shape[0])

        existing_lora_r = getattr(linear, "r", getattr(linear, "lora_r", 0))
        eff_lora_r = existing_lora_r if existing_lora_r > 0 else lora_r
        eff_lora_alpha = getattr(linear, "lora_alpha", getattr(linear, "alpha", lora_alpha))

        layer = cls(
            in_features=in_features,
            out_features=out_features,
            weight=weight,
            bias=b,
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


@COMPRESSOR_REGISTRY.register("mlorq")
@COMPRESSOR_REGISTRY.register("mlorq_compressor")
class MLoRQCompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    MLoRQ Compressor: Multi-Layer Optimal Rank and Quantization under Memory Budget.

    Production hardening features:
    - Relative Frobenius reconstruction error validation with configurable max_error (default 0.50).
    - Safe fallback to original weights by default (fallback_to_original=True).
    - Robust Conv1D detection and support across GPT-2 architectures.
    - Unified critical-layer protection via should_compress_linear.
    - GQA-aware skipping to preserve key-value cache alignment.
    - Tied / shared weight protection tracking parameter IDs.
    - Trainable LoRA adapter integration with base weights frozen.
    - FP32 numerical stability with NaN/Inf guards.
    - Post-compression forward sanity check under torch.no_grad().
    - Comprehensive per-layer stats and error tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        total_memory_budget_mb: float = 500,
        bits_candidates: Optional[List[int]] = None,
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
        super().__init__(model, name="mlorq", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            total_memory_budget_mb = _cfg_get(
                cfg, "mlorq_budget_mb", _cfg_get(cfg, "total_memory_budget_mb", total_memory_budget_mb)
            )
            bits_candidates = _cfg_get(
                cfg, "mlorq_bits_candidates", _cfg_get(cfg, "bits_candidates", bits_candidates)
            )
            max_error = _cfg_get(cfg, "mlorq_max_error", _cfg_get(cfg, "max_error", max_error))
            fallback_to_original = _cfg_get(
                cfg, "mlorq_fallback_to_original", _cfg_get(cfg, "fallback_to_original", fallback_to_original)
            )
            lora_r = _cfg_get(cfg, "lora_r", lora_r)
            lora_alpha = _cfg_get(cfg, "lora_alpha", lora_alpha)
            skip_attention_qkv = _cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv)
            target_attention_only = _cfg_get(cfg, "target_attention_only", target_attention_only)
            min_rank = _cfg_get(cfg, "min_rank", min_rank)

        self.total_budget = float(total_memory_budget_mb) * 8 * 1024 * 1024  # bits
        self.bits_candidates = bits_candidates or [4, 8]
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
        self.layer_bits: Dict[str, int] = {}
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

    @staticmethod
    def _quantize(tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Quantize a float tensor to uniform fixed-point representation."""
        if bits >= 32:
            return tensor.float()
        t = tensor.float()
        min_v, max_v = t.min(), t.max()
        if max_v == min_v:
            return t
        scale = (max_v - min_v) / (2**bits - 1)
        return torch.round((t - min_v) / scale) * scale + min_v

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress model linear layers via MLoRQ."""
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        self.layer_bits = {}
        seen_weights: Set[int] = set()

        total_orig_params = sum(p.numel() for p in self.model.parameters())
        total_comp_params = 0

        layers = []
        for name, module in list(self.model.named_modules()):
            if self._is_target_module(name, module):
                target_layer = getattr(module, "base_layer", module)
                if not hasattr(target_layer, "weight") or target_layer.weight is None:
                    for p in module.parameters(recurse=False):
                        total_comp_params += p.numel()
                    continue

                w_id = id(target_layer.weight)
                if w_id in seen_weights:
                    self.skipped_layers.append(name)
                    for p in module.parameters(recurse=False):
                        total_comp_params += p.numel()
                    continue
                seen_weights.add(w_id)

                raw_w = target_layer.weight.data
                cls_name = target_layer.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(target_layer, "nf")
                W = raw_w.t().contiguous().clone().float() if is_conv1d else raw_w.clone().float()
                out_f, in_f = W.shape

                layers.append({
                    "name": name,
                    "module": module,
                    "target_layer": target_layer,
                    "shape": (out_f, in_f),
                    "W": W,
                    "device": raw_w.device,
                    "dtype": raw_w.dtype,
                    "is_conv1d": is_conv1d,
                })
            else:
                for p in module.parameters(recurse=False):
                    total_comp_params += p.numel()

        total_cost = 0.0
        for info in layers:
            name = info["name"]
            module = info["module"]
            target_layer = info["target_layer"]
            W_orig = info["W"]
            out_f, in_f = info["shape"]
            max_rank = min(out_f, in_f)

            best = None
            for bits in sorted(self.bits_candidates, reverse=True):
                for rank in range(max_rank, max(0, self.min_rank - 1), -max(1, max_rank // 8)):
                    mem = (out_f * rank + rank * in_f) * bits
                    if total_cost + mem <= self.total_budget:
                        best = (rank, bits, mem)
                        break
                if best is not None:
                    break

            if best is None:
                best = (max(1, min(self.min_rank, max_rank)), min(self.bits_candidates), 0)

            r, b, mem = best
            total_cost += mem

            try:
                U, S, Vt = torch.linalg.svd(W_orig, full_matrices=False)
                r_actual = min(r, len(S))
                U_r = U[:, :r_actual]
                S_r = S[:r_actual]
                Vt_r = Vt[:r_actual, :]

                # Quantize factors
                U_q = self._quantize(U_r, b).to(info["device"])
                S_q = self._quantize(S_r, b).to(info["device"])
                Vt_q = self._quantize(Vt_r, b).to(info["device"])

                W_new = U_q @ torch.diag(S_q) @ Vt_q
                W_new = torch.nan_to_num(W_new, nan=0.0)

                w_norm = torch.norm(W_orig, p="fro").item()
                err = (torch.norm(W_orig - W_new, p="fro") / max(w_norm, 1e-12)).item()
                self.layer_errors[name] = float(err)
                self.layer_ranks[name] = r_actual
                self.layer_bits[name] = b

                if not math.isfinite(err) or err > self.max_error:
                    logger.warning(
                        "MLoRQ reconstruction error %.4f on %s exceeded threshold %.4f.",
                        err, name, self.max_error,
                    )
                    if self.fallback_to_original:
                        self.fallback_layers.append(name)
                        for p in module.parameters(recurse=False):
                            p.requires_grad = False
                            total_comp_params += p.numel()
                        continue

                bias = None
                if target_layer.bias is not None:
                    bias = self._quantize(target_layer.bias.data.float(), b).to(dtype=info["dtype"], device=info["device"])

                final_w = W_new.t().contiguous() if info["is_conv1d"] else W_new
                new_layer = MLoRQLoRALinear.from_linear(
                    module,
                    weight=final_w.to(dtype=info["dtype"], device=info["device"]),
                    bias=bias,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )
                replace_module(self.model, name, new_layer)
                self.compressed_layers.append(name)
                for p in new_layer.parameters(recurse=False):
                    total_comp_params += p.numel()
            except Exception as e:
                logger.warning("MLoRQ compression exception on %s: %s", name, e)
                self.fallback_layers.append(name)
                self.layer_errors[name] = 1.0
                for p in module.parameters(recurse=False):
                    p.requires_grad = False
                    total_comp_params += p.numel()

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
            logger.warning("MLoRQ post-compression forward check warning: %s", e)
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
            "layer_bits": self.layer_bits,
            "avg_reconstruction_error": avg_err,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "effective_parameter_ratio": eff_ratio,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "forward_check_success": forward_check_success,
            "method": "mlorq",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("mlorq_factory")
def mlorq_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """MLoRQ compressor factory function."""
    total_budget_mb = _cfg_get(comp_config, "mlorq_budget_mb", kwargs.pop("total_memory_budget_mb", 500))
    bits_candidates = _cfg_get(comp_config, "mlorq_bits_candidates", kwargs.pop("bits_candidates", [4, 8]))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "mlorq_max_error", _cfg_get(comp_config, "max_error", 0.50)))
    fallback = kwargs.pop("fallback_to_original", _cfg_get(
        comp_config, "mlorq_fallback_to_original", _cfg_get(comp_config, "fallback_to_original", True)
    ))
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    target_attention_only = kwargs.pop("target_attention_only", _cfg_get(comp_config, "target_attention_only", False))
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = MLoRQCompressor(
        model,
        total_memory_budget_mb=total_budget_mb,
        bits_candidates=bits_candidates,
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
    "MLoRQLoRALinear",
    "MLoRQCompressor",
    "mlorq_factory",
]
