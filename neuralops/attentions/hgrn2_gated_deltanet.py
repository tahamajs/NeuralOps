"""
HGRN-2 Gated DeltaNet Hybrid — Production-Hardened v3.

FIXES in v3:
  - FIX 1: Trust branch freezing logic (removed destructive loop in Wrapper).
  - FIX 2: Gate is now Token-wise scalar (d_model -> 1) instead of Element-wise (d_model -> d_model).
  - FIX 3: Removed Wrapper output_scale to prevent Double Scaling (Vanishing Gradients).
  All v2 fixes preserved (NaN guards, env var toggles, correct branch fusion).
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch
import torch.nn as nn

from neuralops.attentions.gated_delta_net import GatedDeltaNet, LoRALinear
from neuralops.attentions.hgrn2_gated_layer import HGRN2Layer
from neuralops.core.adapters import UniversalAdapter
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "")
    return v.lower() in ("1", "true", "yes", "on") if v else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _unpack_weight_bias(module: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if module is None:
        return None, None
    base = getattr(module, "base_layer", module)
    w = getattr(base, "weight", None)
    b = getattr(base, "bias", None)
    if w is not None:
        w = w.data.clone()
        cls_name = getattr(module, "__class__", type(module)).__name__
        base_cls = getattr(base, "__class__", type(base)).__name__
        is_conv1d = hasattr(base, "nf") or "Conv1D" in cls_name or "Conv1D" in base_cls
        if is_conv1d and w.ndim == 2:
            w = w.t().contiguous()
    if b is not None:
        b = b.data.clone()
    return w, b


@ATTENTION_REGISTRY.register("hgrn2_gated_deltanet")
class HGRN2GatedDeltaNet(BaseAttention):
    """HGRN-2 + Gated DeltaNet hybrid with proper fusion."""

    is_custom_attention: bool = True

    def __init__(self, config=None, rank=None, d_model=None, num_heads=None,
                 comp_config=None, old_attn=None, lora_rank=16, lora_alpha=32.0,
                 is_decoder=None, **kwargs):
        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        # d_model
        if d_model is None:
            if isinstance(config, int):
                d_model = config
            elif config is not None:
                d_model = (_cfg_get(config, "hidden_size", None)
                           or _cfg_get(config, "d_model", None)
                           or _cfg_get(config, "n_embd", None)
                           or _cfg_get(config, "dim", None))
        if d_model is None and old_attn is not None:
            d_model = (getattr(old_attn, "hidden_size", None)
                       or getattr(old_attn, "embed_dim", None)
                       or getattr(old_attn, "all_head_size", None))
            if d_model is None:
                for cand in ("q_proj", "q_lin", "query", "c_attn"):
                    m = getattr(old_attn, cand, None)
                    if m is not None and hasattr(m, "in_features"):
                        d_model = m.in_features
                        break
        if d_model is None:
            d_model = kwargs.get("hidden_size", kwargs.get("d_model", 768))

        # num_heads
        if num_heads is None and config is not None and not isinstance(config, int):
            num_heads = (_cfg_get(config, "num_attention_heads", None)
                         or _cfg_get(config, "num_heads", None)
                         or _cfg_get(config, "n_heads", None)
                         or _cfg_get(config, "n_head", None))
        if num_heads is None and old_attn is not None:
            num_heads = (getattr(old_attn, "num_heads", None)
                         or getattr(old_attn, "num_attention_heads", None)
                         or getattr(old_attn, "n_heads", None))
        if num_heads is None:
            num_heads = kwargs.get("num_attention_heads", kwargs.get("num_heads", 12))

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)

        # Decoder detection
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            model_type = str(_cfg_get(config, "model_type", "")).lower()
            is_dec = bool(
                _cfg_get(config, "is_decoder", False)
                or _cfg_get(config, "is_causal", False)
                or kwargs.get("is_causal", False)
                or kwargs.get("is_decoder", False)
            )
            _dec_pfx = ("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                        "opt", "bloom", "falcon")
            if any(model_type.startswith(p) for p in _dec_pfx):
                is_dec = True
            if old_attn is not None:
                if getattr(old_attn, "is_decoder", False) or hasattr(old_attn, "c_attn"):
                    is_dec = True
            self.is_decoder = bool(is_dec)

        # LoRA
        self.lora_rank = int(_cfg_get(comp_config, "lora_rank", kwargs.get("lora_rank", lora_rank)))
        self.lora_alpha = float(_cfg_get(comp_config, "lora_alpha", kwargs.get("lora_alpha", lora_alpha)))
        self.freeze_base = self.lora_rank > 0

        # Branch kwargs
        sub_kwargs = {k: v for k, v in kwargs.items()
                      if k not in ("is_decoder", "is_causal", "lora_rank", "lora_alpha")}

        # Branches
        self.hgrn = HGRN2Layer(
            config=config, rank=rank, d_model=self.d_model,
            is_decoder=self.is_decoder, lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha, **sub_kwargs,
        )
        self.deltanet = GatedDeltaNet(
            config=config, rank=rank, d_model=self.d_model, num_heads=self.num_heads,
            is_decoder=self.is_decoder, lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha, **sub_kwargs,
        )

        # FIX 2 (v3): Token-wise scalar routing gate (Linear(D, 1))
        # Initialized to ~98% HGRN, 2% DeltaNet via bias = 4.0
        _gate_bias = _env_float("HGD_GATE_BIAS", 4.0)
        self.gate = nn.Linear(self.d_model, 1)
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.bias.fill_(_gate_bias)

        # FIX 3 (v3): Removed wrapper output_scale completely to prevent Double-Scaling.
        # Branches handle their own output scaling internally.

        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if old_attn is not None:
            self.copy_weights(old_attn)

        target_dtype = kwargs.get("dtype")
        if target_dtype is None and old_attn is not None:
            p = next(old_attn.parameters(), None)
            if p is not None:
                target_dtype = p.dtype
        if target_dtype is not None:
            self.to(target_dtype)

        self._per_layer_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if self._per_layer_trainable > 5e6:
            logger.warning("HGRN2GatedDeltaNet per-layer trainable: %.1fM.",
                           self._per_layer_trainable / 1e6)

    def copy_weights(self, old_attn):
        try:
            for p in old_attn.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        hgrn_ok = False
        deltanet_ok = False
        if hasattr(self.hgrn, "copy_weights"):
            try:
                self.hgrn.copy_weights(old_attn)
                hgrn_ok = True
            except Exception as ex:
                logger.warning("HGRN copy_weights failed: %s", ex)
        if hasattr(self.deltanet, "copy_weights"):
            try:
                self.deltanet.copy_weights(old_attn)
                deltanet_ok = True
            except Exception as ex:
                logger.warning("DeltaNet copy_weights failed: %s", ex)

        # Gate biasing based on which branch successfully copied
        if hgrn_ok and not deltanet_ok:
            with torch.no_grad():
                self.gate.bias.fill_(math.log(0.95 / 0.05))  # ~2.94
            logger.info("Only HGRN copied — biased gate to HGRN")
        elif deltanet_ok and not hgrn_ok:
            with torch.no_grad():
                self.gate.bias.fill_(-math.log(0.95 / 0.05))
            logger.info("Only DeltaNet copied — biased gate to DeltaNet")

        # FIX 1 (v3): Trust the branches to freeze their own base weights!
        if self.freeze_base:
            for branch in (self.hgrn, self.deltanet):
                if hasattr(branch, "freeze_base_weights"):
                    try:
                        branch.freeze_base_weights()
                    except Exception as e:
                        logger.debug("freeze_base_weights on %s failed: %s", type(branch).__name__, e)
            
            # The fusion gate MUST remain trainable to learn the routing
            self.gate.weight.requires_grad = True
            if self.gate.bias is not None:
                self.gate.bias.requires_grad = True

        # Storage Accounting
        try:
            for m in (self.hgrn, self.deltanet):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False,
                output_hidden_states=False, return_dict=True,
                *args, **kwargs):
        
        if encoder_hidden_states is not None:
            raise NotImplementedError("HGRN2GatedDeltaNet does not support cross-attention.")

        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        # Env vars to disable branches for ablation studies
        disable_hgrn = _env_flag("HGD_DISABLE_HGRN", False)
        disable_delta = _env_flag("HGD_DISABLE_DELTANET", False)

        if disable_hgrn and disable_delta:
            raise RuntimeError("Both branches disabled. Nothing to compute.")

        hgrn_out = None
        delta_out = None

        if not disable_hgrn:
            hgrn_res = self.hgrn(hidden_states, attention_mask=attention_mask,
                                  head_mask=head_mask, **kwargs)
            hgrn_out = hgrn_res[0] if isinstance(hgrn_res, tuple) else hgrn_res

        if not disable_delta:
            delta_res = self.deltanet(hidden_states, attention_mask=attention_mask,
                                       head_mask=head_mask, **kwargs)
            delta_out = delta_res[0] if isinstance(delta_res, tuple) else delta_res

        # Fusion Logic
        if hgrn_out is None:
            out = delta_out
        elif delta_out is None:
            out = hgrn_out
        else:
            if hgrn_out.shape != delta_out.shape:
                raise RuntimeError(
                    f"HGRN2GatedDeltaNet branch outputs mismatch: "
                    f"{tuple(hgrn_out.shape)} vs {tuple(delta_out.shape)}."
                )
            gate_dtype = next(self.gate.parameters()).dtype
            # Gamma shape is (B, L, 1), which broadcasts elegantly to (B, L, D)
            gamma = torch.sigmoid(self.gate(hidden_states.to(gate_dtype))).to(hidden_states.dtype)
            out = gamma * hgrn_out.to(hidden_states.dtype) + (1.0 - gamma) * delta_out.to(hidden_states.dtype)

        # Removed Wrapper's output_scale multiplication here to avoid double-damping

        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out, (None if not output_attentions else None)

    def get_ortho_loss(self):
        # Uses branch devices to return zero safely
        p = next(self.parameters(), None)
        dev = p.device if p is not None else torch.device("cpu")
        return torch.zeros((), device=dev)

    def storage_bytes(self):
        def _sz(p):
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0

        out = {}
        for name, m in (("hgrn", self.hgrn), ("deltanet", self.deltanet),
                        ("gate", self.gate)):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# Injection (same as before)
# ===========================================================================
@ATTENTION_REGISTRY.register("hgrn2_gated_deltanet_inject")
@ATTENTION_REGISTRY.register("inject_hgrn2_gated_deltanet")
def inject_hgrn2_gated_deltanet(layer=None, model=None, model_config=None,
                                 comp_config=None, layer_idx=0, total_layers=12,
                                 old_attn=None, lora_rank=16, lora_alpha=32.0,
                                 _depth=0, **kwargs):
    if _depth > 8:
        logger.warning("inject_hgrn2_gated_deltanet: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    layer_list = None
    for path in (("distilbert", "transformer", "layer"),
                 ("roberta", "encoder", "layer"),
                 ("bert", "encoder", "layer"),
                 ("transformer", "h"),
                 ("model", "layers"),
                 ("encoder", "layer"),
                 ("layers",), ("h",)):
        obj = target
        ok = True
        for p in path:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list)):
            layer_list = obj
            break

    if layer_list is not None:
        n = 0
        tot = len(layer_list)
        last = None
        sub_kwargs = {k: v for k, v in kwargs.items() if k not in ("lora_rank", "lora_alpha")}
        for idx, lyr in enumerate(layer_list):
            res = inject_hgrn2_gated_deltanet(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, lora_rank=lora_rank,
                lora_alpha=lora_alpha, _depth=_depth + 1, **sub_kwargs,
            )
            if res is not None:
                n += 1
                last = res
        if n > 0 and last is not None:
            total = n * getattr(last, "_per_layer_trainable", 0)
            if total > 20e6:
                logger.warning("Total trainable across model: %.1fM", total / 1e6)
        return target if n > 0 else None

    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break
    if src_attn is None:
        src_attn = target

    if isinstance(src_attn, HGRN2GatedDeltaNet):
        return None
    if "HGRN2GatedDeltaNet" in src_attn.__class__.__name__:
        return None
    c_name = src_attn.__class__.__name__
    if "DeltaNet" in c_name or "LinearAttention" in c_name:
        return None

    if cfg is None:
        cfg = getattr(src_attn, "config", None) or getattr(target, "config", None)

    try:
        sub_kwargs = {k: v for k, v in kwargs.items() if k not in ("lora_rank", "lora_alpha")}
        new_layer = HGRN2GatedDeltaNet(
            config=cfg, old_attn=src_attn, lora_rank=lora_rank,
            lora_alpha=lora_alpha, **sub_kwargs,
        )
    except Exception as ex:
        logger.warning("HGRN2GatedDeltaNet construction failed: %s", ex)
        return None

    p = next(src_attn.parameters(), None) if src_attn is not target else None
    if p is not None:
        new_layer.to(device=p.device, dtype=p.dtype)

    spliced = False
    if hasattr(target, "attention"):
        if hasattr(target.attention, "self"):
            target.attention.self = new_layer
        else:
            target.attention = new_layer
        spliced = True
    elif hasattr(target, "self"):
        target.self = new_layer
        spliced = True
    elif hasattr(target, "attn"):
        target.attn = new_layer
        spliced = True
    elif hasattr(target, "self_attn"):
        target.self_attn = new_layer
        spliced = True

    if not spliced:
        logger.error("inject_hgrn2_gated_deltanet: no writable attribute on %s",
                     type(target).__name__)
        return None

    return new_layer


__all__ = [
    "HGRN2GatedDeltaNet",
    "inject_hgrn2_gated_deltanet",
]
