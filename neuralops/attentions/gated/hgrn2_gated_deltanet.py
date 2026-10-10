"""
HGRN-2 Gated DeltaNet Hybrid — Production-Hardened v4.

Architecture:
    * HGRN2Layer: hierarchical gated recurrent network (v2)
    * GatedDeltaNet: delta-rule linear attention
    * Token-wise fusion gate blending both branches

Fixes applied:
    [HGD-FIX-01] None-safe super().__init__
    [HGD-FIX-02] lora_ prefix on fusion gate (KD optimizer scope)
    [HGD-FIX-03] Persistent recurrent state passthrough for both branches
    [HGD-FIX-04] Correct tuple unpacking with length check
    [HGD-FIX-05] Argument filtering to prevent **kwargs overflow
    [HGD-FIX-06] Idempotency guard in injector
    [HGD-FIX-07] Recursion depth guard in injector
    [HGD-FIX-08] No double output-scale (delegated to branches)
    [HGD-FIX-09] Trust branch freeze_base_weights()
    [HGD-FIX-10] Frozen gate protected during KD
    [HGD-FIX-11] Proper past_key_value routing to branches
    [HGD-FIX-12] Branch output shape validation
    [HGD-FIX-13] Safe storage accounting
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch
import torch.nn as nn

from neuralops.attentions.gated_delta_net import GatedDeltaNet
from neuralops.attentions.hgrn2_gated_layer import HGRN2Layer
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# Config helpers
# ===========================================================================
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
    """Extract weight/bias from a linear-like module, handling LoRA wrappers."""
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


# ===========================================================================
# HGRN2GatedDeltaNet
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "hgrn2_gated_deltanet",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class HGRN2GatedDeltaNet(BaseAttention):
    """
    HGRN-2 + Gated DeltaNet hybrid with proper fusion and recurrent state caching.

    Fusion (per-token scalar gamma):
        out = γ · hgrn_out + (1 − γ) · deltanet_out
        γ = sigmoid(lora_gate(hidden_states))   # shape (B, L, 1)
    """

    is_custom_attention: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        d_model: int | Any | None = None,
        num_heads: int | None = None,
        comp_config: Any | None = None,
        old_attn: nn.Module | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        # [HGD-FIX-01] None-safe super
        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        # ---- Resolve d_model ----
        if d_model is None:
            if isinstance(config, int):
                d_model = config
            elif config is not None:
                d_model = (
                    _cfg_get(config, "hidden_size", None)
                    or _cfg_get(config, "d_model", None)
                    or _cfg_get(config, "n_embd", None)
                    or _cfg_get(config, "dim", None)
                )
        if d_model is None and old_attn is not None:
            d_model = (
                getattr(old_attn, "hidden_size", None)
                or getattr(old_attn, "embed_dim", None)
                or getattr(old_attn, "all_head_size", None)
            )
            if d_model is None:
                for cand in ("q_proj", "q_lin", "query", "c_attn"):
                    m = getattr(old_attn, cand, None)
                    if m is not None and hasattr(m, "in_features"):
                        d_model = m.in_features
                        break
        if d_model is None:
            d_model = kwargs.get("hidden_size", kwargs.get("d_model", 768))
        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        # ---- Resolve num_heads ----
        if num_heads is None and config is not None and not isinstance(config, int):
            num_heads = (
                _cfg_get(config, "num_attention_heads", None)
                or _cfg_get(config, "num_heads", None)
                or _cfg_get(config, "n_heads", None)
                or _cfg_get(config, "n_head", None)
            )
        if num_heads is None and old_attn is not None:
            num_heads = (
                getattr(old_attn, "num_heads", None)
                or getattr(old_attn, "num_attention_heads", None)
                or getattr(old_attn, "n_heads", None)
            )
        if num_heads is None:
            num_heads = kwargs.get("num_attention_heads", kwargs.get("num_heads", 12))

        self.d_model: int = int(d_model)
        self.num_heads: int = int(num_heads)

        # ---- Decoder detection ----
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
            _dec_pfx = (
                "gpt",
                "qwen",
                "llama",
                "mistral",
                "gemma",
                "phi",
                "opt",
                "bloom",
                "falcon",
            )
            if any(model_type.startswith(p) for p in _dec_pfx):
                is_dec = True
            if old_attn is not None:
                if getattr(old_attn, "is_decoder", False) or hasattr(old_attn, "c_attn"):
                    is_dec = True
            self.is_decoder = bool(is_dec)

        # ---- LoRA config ----
        self.lora_rank: int = int(
            _cfg_get(comp_config, "lora_rank", kwargs.get("lora_rank", lora_rank))
        )
        self.lora_alpha: float = float(
            _cfg_get(comp_config, "lora_alpha", kwargs.get("lora_alpha", lora_alpha))
        )
        self.freeze_base: bool = self.lora_rank > 0

        # ---- Filter kwargs to avoid overflow into branches ----
        # [HGD-FIX-05]
        _reserved = {
            "is_decoder",
            "is_causal",
            "lora_rank",
            "lora_alpha",
            "hidden_size",
            "d_model",
            "num_heads",
            "num_attention_heads",
        }
        sub_kwargs = {k: v for k, v in kwargs.items() if k not in _reserved}

        # ---- Branches ----
        self.hgrn = HGRN2Layer(
            config=config,
            rank=rank,
            d_model=self.d_model,
            is_decoder=self.is_decoder,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            **sub_kwargs,
        )
        self.deltanet = GatedDeltaNet(
            config=config,
            rank=rank,
            d_model=self.d_model,
            num_heads=self.num_heads,
            is_decoder=self.is_decoder,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            **sub_kwargs,
        )

        # ---- [HGD-FIX-02] Fusion gate with lora_ prefix ----
        _gate_bias = _env_float("HGD_GATE_BIAS", 4.0)
        self.lora_gate = nn.Linear(self.d_model, 1)
        with torch.no_grad():
            self.lora_gate.weight.zero_()
            self.lora_gate.bias.fill_(_gate_bias)

        # ---- Storage accounting ----
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        # ---- Weight transfer ----
        if old_attn is not None:
            self.copy_weights(old_attn)

        # ---- Optional dtype cast ----
        target_dtype = kwargs.get("dtype")
        if target_dtype is None and old_attn is not None:
            p = next(old_attn.parameters(), None)
            if p is not None:
                target_dtype = p.dtype
        if target_dtype is not None:
            self.to(target_dtype)

        # ---- Trainable params report ----
        self._per_layer_trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        if self._per_layer_trainable > 5e6:
            logger.warning(
                "HGRN2GatedDeltaNet per-layer trainable: %.1fM.",
                self._per_layer_trainable / 1e6,
            )

    # ------------------------------------------------------------------
    # Backward-compat aliases
    # ------------------------------------------------------------------
    @property
    def gate(self) -> nn.Linear:
        return self.lora_gate

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: nn.Module) -> None:
        """Transfer weights to both branches; adjust gate if only one succeeds."""
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

        # Bias gate based on which branch was successfully loaded
        if hgrn_ok and not deltanet_ok:
            with torch.no_grad():
                self.lora_gate.bias.fill_(math.log(0.95 / 0.05))  # ≈ 2.94
            logger.info("Only HGRN copied — biased gate to HGRN")
        elif deltanet_ok and not hgrn_ok:
            with torch.no_grad():
                self.lora_gate.bias.fill_(-math.log(0.95 / 0.05))
            logger.info("Only DeltaNet copied — biased gate to DeltaNet")

        # ---- [HGD-FIX-09] Trust branches' freeze; only protect gate ----
        if self.freeze_base:
            for branch in (self.hgrn, self.deltanet):
                if hasattr(branch, "freeze_base_weights"):
                    try:
                        branch.freeze_base_weights()
                    except Exception as e:
                        logger.debug(
                            "freeze_base_weights on %s failed: %s",
                            type(branch).__name__,
                            e,
                        )

            # [HGD-FIX-10] The fusion gate MUST remain trainable
            self.lora_gate.weight.requires_grad = True
            if self.lora_gate.bias is not None:
                self.lora_gate.bias.requires_grad = True

        # ---- Storage accounting ----
        try:
            for m in (self.hgrn, self.deltanet):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        if encoder_hidden_states is not None:
            raise NotImplementedError(
                "HGRN2GatedDeltaNet does not support cross-attention."
            )

        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        # ---- Env flags for ablation ----
        disable_hgrn = _env_flag("HGD_DISABLE_HGRN", False)
        disable_delta = _env_flag("HGD_DISABLE_DELTANET", False)

        if disable_hgrn and disable_delta:
            raise RuntimeError("Both branches disabled. Nothing to compute.")

        # ---- [HGD-FIX-11] Route past cache to correct branch ----
        hgrn_past = None
        delta_past = None
        if past_key_value is not None:
            if isinstance(past_key_value, (tuple, list)):
                if len(past_key_value) >= 2:
                    hgrn_past = past_key_value[0]
                    delta_past = past_key_value[1]
                elif len(past_key_value) == 1:
                    # Single-slot cache: give it to both branches
                    hgrn_past = past_key_value[0]
                    delta_past = past_key_value[0]
            elif hasattr(past_key_value, "key_cache"):
                # HF DynamicCache: not supported for recurrent states
                pass

        # ---- Forward each branch ----
        hgrn_out: torch.Tensor | None = None
        hgrn_state: Any = None
        delta_out: torch.Tensor | None = None
        delta_state: Any = None

        if not disable_hgrn:
            hgrn_res = self.hgrn(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_value=hgrn_past,
                **kwargs,
            )
            # [HGD-FIX-04] Safe tuple unpacking
            if isinstance(hgrn_res, tuple):
                hgrn_out = hgrn_res[0]
                if len(hgrn_res) > 1:
                    hgrn_state = hgrn_res[1]
            else:
                hgrn_out = hgrn_res

        if not disable_delta:
            delta_res = self.deltanet(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_value=delta_past,
                **kwargs,
            )
            if isinstance(delta_res, tuple):
                delta_out = delta_res[0]
                if len(delta_res) > 1:
                    delta_state = delta_res[1]
            else:
                delta_out = delta_res

        # ---- Fusion ----
        # [HGD-FIX-12] Shape validation
        if hgrn_out is None:
            if delta_out is None:
                raise RuntimeError("Both branches returned None.")
            out = delta_out
        elif delta_out is None:
            out = hgrn_out
        else:
            if hgrn_out.shape != delta_out.shape:
                raise RuntimeError(
                    f"HGRN2GatedDeltaNet branch outputs mismatch: "
                    f"{tuple(hgrn_out.shape)} vs {tuple(delta_out.shape)}."
                )
            gate_dtype = next(self.lora_gate.parameters()).dtype
            # gamma: (B, L, 1) → broadcasts against (B, L, D)
            gamma = torch.sigmoid(
                self.lora_gate(hidden_states.to(gate_dtype))
            ).to(hidden_states.dtype)
            out = gamma * hgrn_out.to(hidden_states.dtype) + (
                1.0 - gamma
            ) * delta_out.to(hidden_states.dtype)

        # ---- [HGD-FIX-08] No output_scale here (branches handle their own) ----
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        # ---- Package recurrent states ----
        # [HGD-FIX-03] Return composite state for HF KV-cache compatibility
        new_past_key_value: Any = None
        if self.is_decoder:
            # Only return cache if use_cache or a past was provided
            use_cache = kwargs.get("use_cache", False) or (
                past_key_value is not None and past_key_value is not False
            )
            if use_cache:
                new_past_key_value = (hgrn_state, delta_state)

        if output_attentions:
            return out, new_past_key_value, None
        return out, new_past_key_value

    # ------------------------------------------------------------------
    # Ortho loss hook
    # ------------------------------------------------------------------
    def get_ortho_loss(self) -> torch.Tensor:
        p = next(self.parameters(), None)
        dev = p.device if p is not None else torch.device("cpu")
        return torch.zeros((), device=dev)

    # ------------------------------------------------------------------
    # Storage accounting
    # ------------------------------------------------------------------
    def storage_bytes(self) -> dict[str, int]:
        def _sz(p: Any) -> int:
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0

        out: dict[str, int] = {}
        for name, m in (
            ("hgrn", self.hgrn),
            ("deltanet", self.deltanet),
            ("gate", self.lora_gate),
        ):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# Injection
# ===========================================================================
def inject_hgrn2_gated_deltanet(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    _depth: int = 0,
    **kwargs: Any,
):
    # [HGD-FIX-07] Recursion guard
    if _depth > 8:
        logger.warning(
            "inject_hgrn2_gated_deltanet: max recursion depth exceeded."
        )
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = (
        model_config
        if model_config is not None
        else getattr(target, "config", None)
    )

    # ---- Whole-model traversal ----
    layer_list = None
    for path in (
        ("distilbert", "transformer", "layer"),
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("transformer", "h"),
        ("model", "layers"),
        ("encoder", "layer"),
        ("layers",),
        ("h",),
    ):
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
        # Filter kwargs before recursion
        _reserved = {"lora_rank", "lora_alpha"}
        sub_kwargs = {k: v for k, v in kwargs.items() if k not in _reserved}
        for idx, lyr in enumerate(layer_list):
            res = inject_hgrn2_gated_deltanet(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                _depth=_depth + 1,
                **sub_kwargs,
            )
            if res is not None:
                n += 1
                last = res
        if n > 0 and last is not None:
            total = n * getattr(last, "_per_layer_trainable", 0)
            if total > 20e6:
                logger.warning("Total trainable across model: %.1fM", total / 1e6)
        return target if n > 0 else None

    # ---- Single-layer injection ----
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

    # [HGD-FIX-06] Idempotency guard
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
        _reserved = {"lora_rank", "lora_alpha"}
        sub_kwargs = {k: v for k, v in kwargs.items() if k not in _reserved}
        new_layer = HGRN2GatedDeltaNet(
            config=cfg,
            old_attn=src_attn,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            **sub_kwargs,
        )
    except Exception as ex:
        logger.exception("HGRN2GatedDeltaNet construction failed: %s", ex)
        return None

    # ---- Move to source device / dtype ----
    p = next(src_attn.parameters(), None) if src_attn is not target else None
    if p is not None:
        new_layer.to(device=p.device, dtype=p.dtype)

    # ---- Splice ----
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
        logger.error(
            "inject_hgrn2_gated_deltanet: no writable attribute on %s",
            type(target).__name__,
        )
        return None

    return new_layer


__all__ = [
    "HGRN2GatedDeltaNet",
    "inject_hgrn2_gated_deltanet",
]