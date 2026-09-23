"""
FAQ (Feature-Activation-Aware Quantization): adjusts quantization scales using
activation statistics. Registered as 'faq'.

Bug-fix map:
  FQ-FIX-01  LoRA dtype matches base weight dtype.
  FQ-FIX-02  weight / bias mutation under torch.no_grad().
  FQ-FIX-03  act_scale registered as non-persistent buffer.
  FQ-FIX-04  forward sanity check includes attention_mask + correct log message.
  FQ-FIX-05  effective_compression_rate reflects actual quantized layers.
  FQ-FIX-06  multiple calibration batches supported.
  FQ-FIX-07  bias quantization consistent with weight (use_ste flags).
  FQ-FIX-08  LoRA scaling respected (alpha / r).
  FQ-FIX-09  fallback path skips LoRA mutation correctly.
  FQ-FIX-10  _compressed_params initialized in __init__.
  FQ-FIX-11  from_linear act_scale optional and shape-safe.
  FQ-FIX-12  covariance uses per-feature E[x^2] (diagonal), not full covariance.
  FQ-FIX-13  DataLoader support in calibration normalization.
  FQ-FIX-14  double-injection guard via _injected flag.
  FQ-FIX-15  calibration normalization helper added.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.compression_engine.astro_quantizer import (
    STEFakeQuantize,
    quantize_dequantize_tensor,
)
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


# ===========================================================================
# FAQLinear
# ===========================================================================

class FAQLinear(nn.Module):
    is_custom_attention: bool = True
    """Linear with FAQ activation-aware quantization, STE backprop and optional LoRA."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 8,
        symmetric: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features  # Conv1D compat
        self.bits = bits
        self.symmetric = symmetric
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        requires_grad = not (self.lora_r > 0)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype),
            requires_grad=requires_grad,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype),
                requires_grad=requires_grad,
            )
        else:
            self.register_parameter("bias", None)

        self.register_buffer(
            "act_scale",
            torch.ones(in_features, device=device, dtype=dtype),
            persistent=False,      # FQ-FIX-03
        )
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # FQ-FIX-01: LoRA in base dtype.
        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_r, in_features), dtype=dtype, device=device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((out_features, self.lora_r), dtype=dtype, device=device)
            )
            with torch.no_grad():
                tmp = self.lora_A.data.float()
                nn.init.kaiming_uniform_(tmp, a=5 ** 0.5)
                self.lora_A.data.copy_(tmp.to(dtype))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @classmethod
    def from_linear(
        cls,
        module: Union[nn.Linear, Any],
        act_scale: Optional[torch.Tensor] = None,   # FQ-FIX-11: optional
        bits: int = 8,
        symmetric: bool = True,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> "FAQLinear":
        target_mod = getattr(module, "base_layer", module)
        cls_name = target_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        raw_w = target_mod.weight.data
        if is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.t().contiguous()
        else:
            out_f, in_f = raw_w.shape
            W = raw_w

        eff_lora_r = lora_r
        eff_lora_alpha = lora_alpha
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            r_attr = getattr(module, "r", None)
            if r_attr is not None and int(r_attr) > 0:
                eff_lora_r = int(r_attr)
            else:
                la = getattr(module, "lora_A", None)
                if la is not None and hasattr(la, "weight") and la.weight is not None:
                    eff_lora_r = int(la.weight.shape[0])
            eff_lora_alpha = float(getattr(module, "lora_alpha", eff_lora_alpha))

        layer = cls(
            in_features=in_f,
            out_features=out_f,
            bias=(getattr(target_mod, "bias", None) is not None and target_mod.bias is not None),
            bits=bits,
            symmetric=symmetric,
            lora_r=eff_lora_r,
            lora_alpha=eff_lora_alpha,
            device=W.device,
            dtype=W.dtype,
        )
        with torch.no_grad():
            layer.weight.data.copy_(W)
            if (getattr(target_mod, "bias", None) is not None
                    and target_mod.bias is not None
                    and layer.bias is not None):
                layer.bias.data.copy_(target_mod.bias.data)

            # FQ-FIX-11: act_scale optional and shape-safe.
            if act_scale is not None:
                s = act_scale.detach().view(-1)[:in_f].to(device=W.device, dtype=W.dtype)
                n = min(layer.act_scale.numel(), s.numel())
                layer.act_scale[:n].copy_(s[:n])

            # Copy existing LoRA weights.
            if (hasattr(module, "lora_A") and hasattr(module, "lora_B")
                    and layer.lora_A is not None and layer.lora_B is not None):
                try:
                    la = getattr(module.lora_A, "weight", module.lora_A)
                    lb = getattr(module.lora_B, "weight", module.lora_B)
                    if hasattr(la, "data") and la.data.shape == layer.lora_A.data.shape:
                        layer.lora_A.data.copy_(la.data)
                    if hasattr(lb, "data") and lb.data.shape == layer.lora_B.data.shape:
                        layer.lora_B.data.copy_(lb.data)
                except Exception as e:
                    logger.debug("FAQ LoRA copy failed: %s", e)

        return layer

    def get_weight(self) -> torch.Tensor:
        s = torch.clamp(self.act_scale, min=1e-6).unsqueeze(0).to(
            device=self.weight.device, dtype=self.weight.dtype
        )
        w_scaled = self.weight * s
        w_q = quantize_dequantize_tensor(
            w_scaled,
            bits=self.bits,
            symmetric=self.symmetric,
            per_channel=True,
            use_ste=False,
        )
        return w_q / s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.clamp(self.act_scale, min=1e-6).unsqueeze(0).to(
            device=self.weight.device, dtype=self.weight.dtype
        )
        w_scaled = self.weight * s
        w_q = quantize_dequantize_tensor(
            w_scaled,
            bits=self.bits,
            symmetric=self.symmetric,
            per_channel=True,
            use_ste=self.training,
        )
        w_restored = w_q / s

        b_q = None
        if self.bias is not None:
            # FQ-FIX-07: same STE flag as weight.
            b_q = quantize_dequantize_tensor(
                self.bias,
                bits=self.bits,
                symmetric=self.symmetric,
                per_channel=False,
                use_ste=self.training,
            )

        out = nn.functional.linear(
            x,
            w_restored.to(dtype=x.dtype, device=x.device),
            b_q.to(dtype=x.dtype, device=x.device) if b_q is not None else None,
        )

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            # FQ-FIX-08: LoRA scaling matches PEFT convention.
            lora_out = nn.functional.linear(
                nn.functional.linear(x, self.lora_A), self.lora_B
            )
            out = out + self.scaling * lora_out

        return out


# ===========================================================================
# FAQQuantizer
# ===========================================================================

@COMPRESSOR_REGISTRY.register("faq")
@COMPRESSOR_REGISTRY.register("faq_quantizer")
class FAQQuantizer(BaseCompressor):
    """
    FAQQuantizer: Feature-Activation-Aware Quantization.

    Collects activation statistics to scale channel-wise quantization step sizes,
    minimizing activation-weighted reconstruction distortion.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        bits: int = 8,
        calibration_texts: Optional[List[str]] = None,
        calibration_data: Optional[Any] = None,
        alpha: float = 0.5,
        symmetric: bool = True,
        use_ste: bool = True,
        lora_r: int = 16,
        lora_alpha: float = 1.0,
        max_error: float = 0.35,
        target_attention_only: bool = False,
        fallback_to_original: bool = True,
        skip_attention_qkv: bool = False,
        cfg: Optional[Any] = None,
        num_calibration_batches: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="faq", **kwargs)
        cfg = cfg or kwargs.get("comp_config", None)
        self.tokenizer = tokenizer
        self.bits = int(_cfg_get(cfg, "faq_bits", _cfg_get(cfg, "bits", bits)))
        self.calibration_texts = calibration_texts
        self.calibration_data = calibration_data
        self.alpha = float(_cfg_get(cfg, "faq_alpha", _cfg_get(cfg, "alpha", alpha)))
        self.symmetric = bool(_cfg_get(cfg, "faq_symmetric", _cfg_get(cfg, "symmetric", symmetric)))
        self.use_ste = bool(_cfg_get(cfg, "faq_use_ste", _cfg_get(cfg, "use_ste", use_ste)))
        self.lora_r = int(_cfg_get(cfg, "lora_r", _cfg_get(cfg, "lora_rank", lora_r)))
        self.lora_alpha = float(_cfg_get(cfg, "lora_alpha", lora_alpha))
        self.max_error = float(_cfg_get(cfg, "faq_max_error", _cfg_get(cfg, "max_error", max_error)))
        self.target_attention_only = bool(
            _cfg_get(cfg, "faq_target_attention_only",
                     _cfg_get(cfg, "target_attention_only", target_attention_only))
        )
        self.fallback_to_original = bool(
            _cfg_get(cfg, "faq_fallback_to_original",
                     _cfg_get(cfg, "fallback_to_original", fallback_to_original))
        )
        self.skip_attention_qkv = bool(_cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv))
        self.cfg = cfg
        self.num_calibration_batches = max(1, int(
            _cfg_get(cfg, "num_calibration_batches", num_calibration_batches)
        ))

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self._compressed_params: int = 0   # FQ-FIX-10
        self._injected: bool = False        # FQ-FIX-14

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------
    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads:
            name_lower = name.lower()
            if any(p in name_lower for p in
                   ["q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn"]):
                return True
        return False

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
        is_linear_like = (
            isinstance(target_mod, nn.Linear)
            or "conv1d" in cls_name
            or hasattr(target_mod, "nf")
        )
        if not is_linear_like or isinstance(target_mod, FAQLinear):
            return False

        if not should_compress_linear(self.model, name):
            return False

        if self.target_attention_only:
            last_part = name.split(".")[-1]
            is_attn = (last_part in _ATTN_NAMES) or any(k in name_lower for k in ("attn", "attention"))
            if not is_attn:
                return False

        return True

    # ------------------------------------------------------------------
    # FQ-FIX-12: per-feature E[x^2] accumulation (diagonal of covariance).
    # FQ-FIX-13: DataLoader support.
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_calibration_batches(data: Any) -> List[Any]:
        """FQ-FIX-13: convert DataLoader / list / tensor / dict into a list of batches."""
        if data is None:
            return []
        if isinstance(data, torch.Tensor):
            return [data]
        if isinstance(data, dict):
            return [data]
        if isinstance(data, (list, tuple)):
            return list(data)
        if hasattr(data, "__iter__"):
            return list(data)
        return [data]

    def _collect_activation_scales(self) -> Dict[str, torch.Tensor]:
        """
        FQ-FIX-12: accumulate E[x^2] per input feature.
        This is the diagonal of the input covariance; sufficient for
        per-channel quantization.
        """
        running_sq: Dict[str, torch.Tensor] = {}
        sample_counts: Dict[str, int] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module, inp, outp):
                if not inp or not isinstance(inp[0], torch.Tensor):
                    return
                x = inp[0].detach()
                target_mod = getattr(module, "base_layer", module)
                cls_name = target_mod.__class__.__name__.lower()
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                in_f = (target_mod.weight.shape[0] if is_conv1d
                        else target_mod.weight.shape[1])

                x = x.reshape(-1, in_f).float()
                n = x.shape[0]
                if n == 0:
                    return

                sq = (x * x).sum(dim=0).cpu()   # (in_f,)
                if layer_name not in running_sq:
                    running_sq[layer_name] = sq
                    sample_counts[layer_name] = n
                else:
                    running_sq[layer_name] += sq
                    sample_counts[layer_name] += n
            return hook

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module) and not self._is_gqa_asymmetric(name, module):
                hooks.append(module.register_forward_hook(make_hook(name)))

        self.model.eval()
        first_p = next(self.model.parameters(), None)
        dev = first_p.device if first_p is not None else torch.device("cpu")

        try:
            with torch.no_grad():
                if self.tokenizer is not None and self.calibration_texts:
                    bs = max(1, self.num_calibration_batches)
                    for idx in range(0, len(self.calibration_texts), bs):
                        sub = self.calibration_texts[idx: idx + bs]
                        if not sub:
                            break
                        try:
                            inputs = self.tokenizer(
                                sub, return_tensors="pt", padding=True,
                                truncation=True, max_length=128,
                            )
                            inputs = {k: v.to(dev) for k, v in inputs.items()}
                            self.model(**inputs)
                        except Exception as e:
                            logger.debug("FAQ calibration batch failed: %s", e)

                elif self.calibration_data is not None:
                    batches = self._normalize_calibration_batches(self.calibration_data)
                    for i, batch in enumerate(batches):
                        if i >= self.num_calibration_batches:
                            break
                        try:
                            if isinstance(batch, torch.Tensor):
                                self.model(batch.to(dev))
                            elif isinstance(batch, dict):
                                b = {k: v.to(dev) for k, v in batch.items()
                                     if isinstance(v, torch.Tensor)}
                                self.model(**b)
                            elif isinstance(batch, (list, tuple)):
                                self.model(*[b.to(dev) if isinstance(b, torch.Tensor) else b
                                             for b in batch])
                        except Exception as e:
                            logger.debug("FAQ calibration batch %d failed: %s", i, e)
                else:
                    first_linear = next(
                        (m for m in self.model.modules() if isinstance(m, nn.Linear)), None
                    )
                    if first_linear is not None:
                        dummy = torch.randn(2, 16, first_linear.in_features, device=dev)
                        try:
                            self.model(dummy)
                        except Exception as e:
                            logger.debug("FAQ fallback forward failed: %s", e)
        finally:
            for h in hooks:
                h.remove()

        act_scales: Dict[str, torch.Tensor] = {}
        for name, module in self.model.named_modules():
            target_mod = getattr(module, "base_layer", module)
            cls_name = target_mod.__class__.__name__.lower()
            is_linear_like = (
                isinstance(target_mod, nn.Linear)
                or "conv1d" in cls_name
                or hasattr(target_mod, "nf")
            )
            if is_linear_like and hasattr(target_mod, "weight") and target_mod.weight is not None:
                is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
                in_f = (target_mod.weight.shape[0] if is_conv1d
                        else target_mod.weight.shape[1])
                if name in running_sq and sample_counts.get(name, 0) > 0:
                    mean_sq = running_sq[name] / float(sample_counts[name])
                    channel_scale = torch.sqrt(torch.clamp(mean_sq, min=1e-6))
                    channel_scale = channel_scale / channel_scale.mean().clamp(min=1e-6)
                    act_scales[name] = channel_scale
                else:
                    act_scales[name] = torch.ones(in_f)
        return act_scales

    # ------------------------------------------------------------------
    # Main compress
    # ------------------------------------------------------------------
    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        # FQ-FIX-14: double-injection guard.
        if self._injected:
            logger.warning("FAQQuantizer.compress called twice; returning current stats.")
            try:
                return self.model, self.get_compression_stats()
            except Exception:
                return self.model, {}

        act_scales = self._collect_activation_scales()
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        seen_weights: set = set()

        # Mark non-target attention as skipped when target_attention_only is set.
        if self.target_attention_only:
            for name, module in self.model.named_modules():
                if any(part in name.split(".") for part in ["lora_A", "lora_B", "base_layer"]):
                    continue
                target_mod = getattr(module, "base_layer", module)
                cls_name = target_mod.__class__.__name__.lower()
                is_linear_like = (
                    isinstance(target_mod, nn.Linear)
                    or "conv1d" in cls_name
                    or hasattr(target_mod, "nf")
                )
                if is_linear_like and not self._is_target_module(name, module):
                    for p in module.parameters():
                        p.requires_grad = False
                    self.skipped_layers.append(name)

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

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
            W_orig = raw_w.t().float() if is_conv1d else raw_w.float()
            in_f = W_orig.shape[1]

            s_act = act_scales.get(name, torch.ones(in_f)).to(raw_w.device)
            s_pow = torch.pow(torch.clamp(s_act, min=1e-6), self.alpha).unsqueeze(0)

            try:
                if self.use_ste or self.lora_r > 0:
                    faq_layer = FAQLinear.from_linear(
                        module,
                        act_scale=s_act,
                        bits=self.bits,
                        symmetric=self.symmetric,
                        lora_r=self.lora_r,
                        lora_alpha=self.lora_alpha,
                    )
                    W_recon = faq_layer.get_weight().float()
                    W_norm = torch.norm(W_orig, p="fro").clamp_min(1e-8)
                    err = float(torch.norm(W_orig - W_recon, p="fro").item() / W_norm)
                    self.layer_errors[name] = err

                    if err > self.max_error and self.fallback_to_original:
                        self.fallback_layers.append(name)
                        for p in module.parameters():
                            p.requires_grad = False
                        continue

                    replace_module(self.model, name, faq_layer)
                    self.compressed_layers.append(name)
                else:
                    W_scaled = W_orig * s_pow
                    W_q = quantize_dequantize_tensor(
                        W_scaled,
                        bits=self.bits,
                        symmetric=self.symmetric,
                        per_channel=True,
                        use_ste=False,
                    )
                    W_recon = W_q / s_pow
                    W_norm = torch.norm(W_orig, p="fro").clamp_min(1e-8)
                    err = float(torch.norm(W_orig - W_recon, p="fro").item() / W_norm)
                    self.layer_errors[name] = err

                    if err > self.max_error and self.fallback_to_original:
                        self.fallback_layers.append(name)
                        for p in module.parameters():
                            p.requires_grad = False
                        continue

                    # FQ-FIX-02: no_grad mutation.
                    with torch.no_grad():
                        if is_conv1d:
                            target_mod.weight.data.copy_(W_recon.t().to(dtype=raw_w.dtype))
                        else:
                            target_mod.weight.data.copy_(W_recon.to(dtype=raw_w.dtype))

                        if getattr(target_mod, "bias", None) is not None and target_mod.bias is not None:
                            b_q = quantize_dequantize_tensor(
                                target_mod.bias.data,
                                bits=self.bits,
                                symmetric=self.symmetric,
                                per_channel=False,
                                use_ste=False,
                            )
                            target_mod.bias.data.copy_(b_q.to(dtype=target_mod.bias.dtype))

                        if not hasattr(target_mod, "act_scale"):
                            target_mod.register_buffer(
                                "act_scale",
                                s_act.to(dtype=raw_w.dtype, device=raw_w.device),
                                persistent=False,
                            )

                    # Preserve LoRA trainability.
                    if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                        if hasattr(module.lora_A, "weight") and module.lora_A.weight is not None:
                            module.lora_A.weight.requires_grad = True
                        if hasattr(module.lora_B, "weight") and module.lora_B.weight is not None:
                            module.lora_B.weight.requires_grad = True

                    self.compressed_layers.append(name)
            except Exception as e:
                logger.warning("FAQ quantization failed on %s: %s", name, e)
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False

        # FQ-FIX-04: forward sanity check with attention_mask.
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                dummy = torch.randint(0, 100, (1, 8), device=dev, dtype=torch.long)
                attn = torch.ones_like(dummy)
                try:
                    self.model(input_ids=dummy, attention_mask=attn)
                except Exception:
                    try:
                        self.model(dummy)
                    except Exception:
                        first_linear = next(
                            (m for m in self.model.modules()
                             if isinstance(m, (nn.Linear, FAQLinear))), None,
                        )
                        in_dim = getattr(first_linear, "in_features", 8)
                        first_p = next(self.model.parameters(), None)
                        self.model(torch.randn(
                            1, 8, in_dim, device=dev,
                            dtype=first_p.dtype if first_p is not None else torch.float32,
                        ))
        except Exception as e:
            logger.debug("FAQ forward sanity check failed: %s", e)
            forward_check_success = False

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        avg_err = (sum(self.layer_errors.values()) / len(self.layer_errors)) if self.layer_errors else 0.0

        # FQ-FIX-05: effective compression reflects actual quantized layers.
        num_targets = len(self.compressed_layers) + len(self.fallback_layers)
        if num_targets > 0 and self.bits > 0:
            quantized_fraction = len(self.compressed_layers) / num_targets
            fp16_to_int_ratio = 16.0 / float(self.bits)
            effective_compression_rate = 1.0 + quantized_fraction * (fp16_to_int_ratio - 1.0)
        else:
            effective_compression_rate = 1.0

        stats.update({
            "compressed_layers": self.compressed_layers,
            "compressed_layer_count": len(self.compressed_layers),
            "fallback_layers": self.fallback_layers,
            "fallback_layer_count": len(self.fallback_layers),
            "skipped_layers": self.skipped_layers,
            "skipped_layer_count": len(self.skipped_layers),
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            "layer_errors": self.layer_errors,
            "avg_reconstruction_error": avg_err,
            "quantization_bits": self.bits,
            "alpha": self.alpha,
            "symmetric": self.symmetric,
            "use_ste": self.use_ste,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "target_attention_only": self.target_attention_only,
            "forward_check_success": forward_check_success,
            "effective_compression_rate": round(effective_compression_rate, 2),
            "theoretical_vram_saving": f"{(1.0 - self.bits / 16.0) * 100:.1f}% vs FP16",
            "method": "faq_quantization",
        })

        # FQ-FIX-14: mark as injected.
        self._injected = True
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================

@COMPRESSOR_REGISTRY.register("faq_factory")
@COMPRESSOR_REGISTRY.register("faq_quantizer_factory")
def faq_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """FAQ compressor factory function."""
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    bits = kwargs.pop("bits", _cfg_get(comp_config, "faq_bits", _cfg_get(comp_config, "bits", 8)))
    alpha = kwargs.pop("alpha", _cfg_get(comp_config, "faq_alpha", _cfg_get(comp_config, "alpha", 0.5)))
    symmetric = kwargs.pop(
        "symmetric",
        _cfg_get(comp_config, "faq_symmetric",
                 _cfg_get(comp_config, "symmetric", True)),
    )
    use_ste = kwargs.pop(
        "use_ste",
        _cfg_get(comp_config, "faq_use_ste",
                 _cfg_get(comp_config, "use_ste", True)),
    )
    lora_r = kwargs.pop(
        "lora_r",
        _cfg_get(comp_config, "lora_r",
                 _cfg_get(comp_config, "lora_rank", 16)),
    )
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(comp_config, "faq_max_error",
                 _cfg_get(comp_config, "max_error", 0.35)),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(comp_config, "faq_target_attention_only",
                 _cfg_get(comp_config, "target_attention_only", False)),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(comp_config, "faq_fallback_to_original",
                 _cfg_get(comp_config, "fallback_to_original", True)),
    )
    tokenizer = kwargs.pop("tokenizer", getattr(model, "tokenizer", None))
    num_calib = kwargs.pop(
        "num_calibration_batches",
        _cfg_get(comp_config, "num_calibration_batches", 8),
    )

    compressor = FAQQuantizer(
        model,
        tokenizer=tokenizer,
        bits=bits,
        alpha=alpha,
        symmetric=symmetric,
        use_ste=use_ste,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_error=max_error,
        target_attention_only=target_attention_only,
        fallback_to_original=fallback,
        cfg=comp_config,
        num_calibration_batches=num_calib,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "FAQLinear",
    "FAQQuantizer",
    "faq_factory",
]