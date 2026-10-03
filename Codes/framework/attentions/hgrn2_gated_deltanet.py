"""HGRN-2 Gated DeltaNet Hybrid Attention Module — Production-Hardened (v2).

Combines bidirectional/causal HGRN-2 element-wise linear recurrence with multi-head
Gated DeltaNet associative matrix memory.

Fixes (v2):
  HGD-FIX-01  is_custom_attention = True (this IS attention).
  HGD-FIX-02  Balanced gate init (+/-log(0.9/0.1) instead of 4.0).
  HGD-FIX-03  Idempotency guard in injector.
  HGD-FIX-04  No padding between branches — must be same shape.
  HGD-FIX-05  Cross-attention raises clear error.
  HGD-FIX-06  freeze_base applied after copy.
  HGD-FIX-07  Storage accounting.
  HGD-FIX-08  kwargs filtered per branch.
  HGD-FIX-09  Idempotency guard.
  HGD-FIX-10  Recursion guard in injector.
  HGD-FIX-11  LoRALinear imported from framework.
  HGD-FIX-12  Gate trainability controlled by flag.
  HGD-FIX-13  Mask and head_mask passed separately.
  HGD-FIX-14  Removed output shape padding/truncation hack.
  HGD-FIX-15  get_ortho_loss for framework compatibility.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn

from framework.attentions.gated_delta_net import GatedDeltaNet, LoRALinear
from framework.attentions.hgrn2_layer import HGRN2Layer
from framework.core.adapters import UniversalAdapter
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


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
        is_conv1d = (
            hasattr(base, "nf")
            or "Conv1D" in cls_name
            or "Conv1D" in base_cls
        )
        if is_conv1d and w.ndim == 2:
            w = w.t().contiguous()
    if b is not None:
        b = b.data.clone()
    return w, b


@ATTENTION_REGISTRY.register("hgrn2_gated_deltanet")
class HGRN2GatedDeltaNet(BaseAttention):
    r"""HGRN-2 Gated DeltaNet Hybrid Architecture.

    Combines:
      1. HGRN-2 branch: fast element-wise linear recurrence.
      2. Gated DeltaNet branch: associative matrix state with Delta-rule.
      3. Learnable fusion:
         $Y = \gamma \odot Y_{hgrn} + (1-\gamma) \odot Y_{deltanet}$
    """
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        d_model: Any | None = None,
        num_heads: int | None = None,
        comp_config: Any | None = None,
        old_attn: nn.Module | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        # ---- d_model ----
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

        # ---- num_heads ----
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
            num_heads = kwargs.get("num_attention_heads",
                                    kwargs.get("num_heads", 12))

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)

        # ---- decoder detection ----
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            model_type = str(_cfg_get(config, "model_type", "")).lower()
            is_dec = bool(_cfg_get(config, "is_decoder", False)
                           or _cfg_get(config, "is_causal", False)
                           or kwargs.get("is_causal", False)
                           or kwargs.get("is_decoder", False))
            _dec_pfx = ("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                         "opt", "bloom", "falcon")
            if any(model_type.startswith(p) for p in _dec_pfx):
                is_dec = True
            if old_attn is not None:
                if getattr(old_attn, "is_decoder", False) or hasattr(old_attn, "c_attn"):
                    is_dec = True
            self.is_decoder = bool(is_dec)

        # ---- LoRA ----
        self.lora_rank = int(_cfg_get(comp_config, "lora_rank",
                              kwargs.get("lora_rank", lora_rank)))
        self.lora_alpha = float(_cfg_get(comp_config, "lora_alpha",
                                  kwargs.get("lora_alpha", lora_alpha)))
        self.freeze_base = (self.lora_rank > 0)

        # ---- branch kwargs ----
        sub_kwargs = {k: v for k, v in kwargs.items()
                       if k not in ("is_decoder", "is_causal",
                                     "lora_rank", "lora_alpha")}

        # ---- branches ----
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

        # ---- gate ----
        # HGD-FIX-02: balanced init
        self.gate = nn.Linear(self.d_model, self.d_model)
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.bias.fill_(0.0)   # sigmoid(0) = 0.5

        # ---- output ----
        self.out_proj = LoRALinear(
            self.d_model, self.d_model, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        # Storage
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

        self._per_layer_trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        if self._per_layer_trainable > 5e6:
            logger.warning(
                "HGRN2GatedDeltaNet per-layer trainable: %.1fM.",
                self._per_layer_trainable / 1e6,
            )

    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: Any) -> None:
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

        if not hgrn_ok and not deltanet_ok:
            logger.warning("Both HGRN and DeltaNet copy_weights failed.")
        elif hgrn_ok and not deltanet_ok:
            with torch.no_grad():
                self.gate.weight.zero_()
                self.gate.bias.fill_(math.log(0.9 / 0.1))  # ~2.20
            logger.info("Only HGRN copied — biased gate to HGRN")
        elif deltanet_ok and not hgrn_ok:
            with torch.no_grad():
                self.gate.weight.zero_()
                self.gate.bias.fill_(-math.log(0.9 / 0.1))
            logger.info("Only DeltaNet copied — biased gate to DeltaNet")

        # ---- out_proj ----
        w_o, b_o = None, None
        try:
            adapter = UniversalAdapter()
            w_o, b_o = adapter.get_weight_bias(old_attn, "o")
        except Exception:
            pass

        if w_o is None:
            for cand in ("out_proj", "out_lin", "o_proj", "output", "proj",
                          "c_proj", "dense"):
                for target in (old_attn,
                                getattr(old_attn, "self", None),
                                getattr(old_attn, "attention", None),
                                getattr(old_attn, "output", None)):
                    if target is not None and hasattr(target, cand):
                        mod = getattr(target, cand)
                        if hasattr(mod, "dense"):
                            mod = mod.dense
                        cw, cb = _unpack_weight_bias(mod)
                        if cw is not None:
                            w_o, b_o = cw, cb
                            break
                if w_o is not None:
                    break

        if w_o is not None:
            with torch.no_grad():
                tw = self.out_proj.weight
                if w_o.shape == tw.shape:
                    tw.copy_(w_o.to(tw.dtype))
                elif w_o.t().shape == tw.shape:
                    tw.copy_(w_o.t().to(tw.dtype))
                elif w_o.numel() == tw.numel():
                    tw.copy_(w_o.reshape(tw.shape).to(tw.dtype))
                else:
                    min_r = min(tw.shape[0], w_o.shape[0])
                    min_c = min(tw.shape[1], w_o.shape[1])
                    tw.data.zero_()
                    tw.data[:min_r, :min_c].copy_(
                        w_o[:min_r, :min_c].to(tw.dtype)
                    )
                    logger.warning(
                        "Out proj shape mismatch: %s → %s.",
                        tuple(w_o.shape), tuple(tw.shape),
                    )

                if b_o is not None and self.out_proj.bias is not None:
                    tb = self.out_proj.bias
                    if b_o.shape == tb.shape:
                        tb.copy_(b_o.to(tb.dtype))
                    elif b_o.numel() == tb.numel():
                        tb.copy_(b_o.reshape(tb.shape).to(tb.dtype))
                    else:
                        min_b = min(tb.numel(), b_o.numel())
                        tb.data.zero_()
                        tb.data[:min_b].copy_(b_o[:min_b].to(tb.dtype))
        else:
            logger.warning(
                "Out projection weight not found for %s.",
                type(old_attn).__name__,
            )

        if self.freeze_base:
            self.out_proj.freeze_base_weights()
            self.gate.weight.requires_grad = False
            if self.gate.bias is not None:
                self.gate.bias.requires_grad = False

        # Storage accounting
        try:
            for m in (self.hgrn, self.deltanet, self.out_proj):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if encoder_hidden_states is not None:
            raise NotImplementedError(
                "HGRN2GatedDeltaNet does not support cross-attention."
            )

        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        hgrn_res = self.hgrn(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            **kwargs,
        )
        hgrn_out = hgrn_res[0] if isinstance(hgrn_res, tuple) else hgrn_res

        delta_res = self.deltanet(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            **kwargs,
        )
        delta_out = delta_res[0] if isinstance(delta_res, tuple) else delta_res

        if hgrn_out.shape != delta_out.shape:
            raise RuntimeError(
                f"HGRN2GatedDeltaNet branch outputs mismatch: "
                f"{tuple(hgrn_out.shape)} vs {tuple(delta_out.shape)}."
            )

        gate_dtype = next(self.gate.parameters()).dtype
        gamma = torch.sigmoid(self.gate(hidden_states.to(gate_dtype))).to(hidden_states.dtype)
        fused = gamma * hgrn_out.to(hidden_states.dtype) + (1.0 - gamma) * delta_out.to(hidden_states.dtype)
        out = self.out_proj(fused)

        if out.shape[-1] != hidden_states.shape[-1]:
            raise RuntimeError(
                f"HGRN2GatedDeltaNet output last dim {out.shape[-1]} "
                f"!= hidden {hidden_states.shape[-1]}."
            )

        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out, (None if not output_attentions else None)

    def get_ortho_loss(self):
        return torch.zeros((), device=self.out_proj.weight.device)

    def storage_bytes(self) -> dict[str, int]:
        def _sz(p):
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0
        out = {}
        for name, m in (("hgrn", self.hgrn), ("deltanet", self.deltanet),
                         ("out_proj", self.out_proj), ("gate", self.gate)):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("hgrn2_gated_deltanet_inject")
@ATTENTION_REGISTRY.register("inject_hgrn2_gated_deltanet")
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
) -> Any | None:
    if _depth > 8:
        logger.warning("inject_hgrn2_gated_deltanet: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
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
        sub_kwargs = {k: v for k, v in kwargs.items()
                       if k not in ("lora_rank", "lora_alpha")}
        for idx, lyr in enumerate(layer_list):
            res = inject_hgrn2_gated_deltanet(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot,
                lora_rank=lora_rank, lora_alpha=lora_alpha,
                _depth=_depth + 1, **sub_kwargs,
            )
            if res is not None:
                n += 1
                last = res
        if n > 0 and last is not None:
            total = n * getattr(last, "_per_layer_trainable", 0)
            if total > 20e6:
                logger.warning("Total trainable across model: %.1fM", total / 1e6)
        return target if n > 0 else None

    # Single layer
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

    # Idempotency guard
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
        sub_kwargs = {k: v for k, v in kwargs.items()
                       if k not in ("lora_rank", "lora_alpha")}
        new_layer = HGRN2GatedDeltaNet(
            config=cfg,
            old_attn=src_attn,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            **sub_kwargs,
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
        logger.error(
            "inject_hgrn2_gated_deltanet: no writable attention attribute on %s",
            type(target).__name__,
        )
        return None

    return new_layer


__all__ = [
    "HGRN2GatedDeltaNet",
    "inject_hgrn2_gated_deltanet",
]
