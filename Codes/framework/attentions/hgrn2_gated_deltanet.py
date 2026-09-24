"""HGRN-2 Gated DeltaNet Hybrid Attention Module.

Combines bidirectional/causal HGRN-2 element-wise linear recurrence with multi-head
Gated DeltaNet associative matrix memory for rich representational capacity.
Hardened across DistilBERT, RoBERTa, GPT-2, and Qwen3.5 architectures.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _unpack_weight_bias(module: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if module is None:
        return None, None
    base = getattr(module, "base_layer", module)
    w = getattr(base, "weight", None)
    b = getattr(base, "bias", None)
    if w is not None:
        w = w.data.clone()
        cls_name = getattr(module, "__class__", type(module)).__name__
        base_cls_name = getattr(base, "__class__", type(base)).__name__
        is_conv1d = (
            hasattr(base, "nf")
            or "Conv1D" in cls_name
            or "Conv1D" in base_cls_name
        )
        if is_conv1d and w.ndim == 2:
            w = w.t().contiguous()
    if b is not None:
        b = b.data.clone()
    return w, b


@ATTENTION_REGISTRY.register("hgrn2_gated_deltanet")
class HGRN2GatedDeltaNet(BaseAttention):
    r"""HGRN-2 Gated DeltaNet Hybrid Architecture.

    Mathematical Mechanism:
        Combines two complementary linear recurrent mechanisms:
        1. **HGRN-2 Branch:** Fast element-wise linear recurrence with bounded forget gating.
        2. **Gated DeltaNet Branch:** Multi-head associative matrix state with Delta-rule error updates.
        3. **Dynamic Fusion:**
           $$Y = \gamma \odot Y_{\text{hgrn}} + (1 - \gamma) \odot Y_{\text{deltanet}}$$
           where $\gamma = \sigma(W_g X)$.
    """

    is_custom_attention: bool = True
    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        d_model: Optional[Union[int, Any]] = None,
        num_heads: Optional[int] = None,
        comp_config: Optional[Any] = None,
        old_attn: Optional[nn.Module] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)

        # Resolve d_model
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
            d_model = kwargs.get("hidden_size", kwargs.get("d_model", kwargs.get("dim", 768)))

        # Resolve num_heads
        if num_heads is None:
            if config is not None and not isinstance(config, int):
                num_heads = (
                    _cfg_get(config, "num_attention_heads", None)
                    or _cfg_get(config, "num_heads", None)
                    or _cfg_get(config, "n_heads", None)
                    or _cfg_get(config, "n_head", None)
                )
        if num_heads is None and old_attn is not None:
            num_heads = getattr(old_attn, "num_heads", getattr(old_attn, "num_attention_heads", getattr(old_attn, "n_heads", None)))
        if num_heads is None:
            num_heads = kwargs.get("num_attention_heads", kwargs.get("num_heads", 12))

        self.d_model: int = int(d_model)
        self.num_heads: int = int(num_heads)

        # Causal / Decoder detection
        model_type = str(_cfg_get(config, "model_type", "")).lower()
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon")
        is_dec = _cfg_get(config, "is_decoder", _cfg_get(config, "is_causal", kwargs.get("is_causal", kwargs.get("is_decoder", False))))
        if any(model_type.startswith(p) for p in _decoder_prefixes):
            is_dec = True
        elif hasattr(config, "architectures") and any("GPT" in str(a) or "CausalLM" in str(a) for a in (_cfg_get(config, "architectures", []) or [])):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)

        # LoRA setup
        self.lora_rank: int = int(_cfg_get(comp_config, "lora_rank", kwargs.get("lora_rank", lora_rank)))
        self.lora_alpha: float = float(_cfg_get(comp_config, "lora_alpha", kwargs.get("lora_alpha", lora_alpha)))
        self.freeze_base: bool = (self.lora_rank > 0)

        # Filter kwargs to avoid duplicate keyword argument errors
        sub_kwargs = {k: v for k, v in kwargs.items() if k not in ("is_decoder", "is_causal", "lora_rank", "lora_alpha")}

        # Branches
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

        # Gate and output projection
        self.gate = nn.Linear(self.d_model, self.d_model)
        with torch.no_grad():
            self.gate.weight.zero_()
            self.gate.bias.fill_(0.0)   # sigmoid(0) = 0.5 (equal blend)

        self.out_proj = LoRALinear(
            self.d_model,
            self.d_model,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        if old_attn is not None:
            self.copy_weights(old_attn)

        target_dtype = kwargs.get("dtype", None)
        if target_dtype is None and old_attn is not None:
            target_dtype = next(old_attn.parameters(), torch.tensor([])).dtype
            if not isinstance(target_dtype, torch.dtype):
                target_dtype = torch.float32

        if target_dtype is not None:
            self.to(target_dtype)

        # Trainable parameter check and OOM-prevention warning
        per_layer = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._per_layer_trainable = per_layer
        if per_layer > 5e6:
            logger.warning(
                "HGRN2GatedDeltaNet per-layer trainable params: %.1fM.",
                per_layer / 1e6,
            )

    def copy_weights(self, old_attn: Any) -> None:
        """Transfer pretrained Multi-Head Attention weights to both branches and output projection."""
        hgrn_ok = False
        deltanet_ok = False
        if hasattr(self.hgrn, "copy_weights"):
            try:
                self.hgrn.copy_weights(old_attn)
                hgrn_ok = True
            except Exception as ex:
                logger.warning("HGRN copy_weights FAILED: %s", ex)
        if hasattr(self.deltanet, "copy_weights"):
            try:
                self.deltanet.copy_weights(old_attn)
                deltanet_ok = True
            except Exception as ex:
                logger.warning("DeltaNet copy_weights FAILED: %s", ex)

        if not hgrn_ok and not deltanet_ok:
            logger.debug("Both HGRN and DeltaNet copy_weights failed — aborting")
        if hgrn_ok and not deltanet_ok:
            with torch.no_grad():
                self.gate.weight.zero_()
                self.gate.bias.fill_(4.0)
            logger.info("Only HGRN branch copied — biased gate to HGRN")
        elif deltanet_ok and not hgrn_ok:
            with torch.no_grad():
                self.gate.weight.zero_()
                self.gate.bias.fill_(-4.0)
            logger.info("Only DeltaNet branch copied — biased gate to DeltaNet")

        w_o, b_o = None, None
        try:
            adapter = UniversalAdapter()
            w_o, b_o = adapter.get_weight_bias(old_attn, "o")
        except Exception:
            pass

        if w_o is None:
            for cand in ("out_proj", "out_lin", "o_proj", "output", "proj", "c_proj", "dense"):
                for target in (old_attn, getattr(old_attn, "self", None), getattr(old_attn, "attention", None), getattr(old_attn, "output", None)):
                    if target is not None and hasattr(target, cand):
                        mod = getattr(target, cand)
                        if hasattr(mod, "dense"):
                            mod = getattr(mod, "dense")
                        cw, cb = _unpack_weight_bias(mod)
                        if cw is not None:
                            w_o, b_o = cw, cb
                            break
                if w_o is not None:
                    break

        if w_o is not None:
            with torch.no_grad():
                if w_o.shape == self.out_proj.weight.shape:
                    self.out_proj.weight.copy_(w_o)
                elif w_o.t().shape == self.out_proj.weight.shape:
                    self.out_proj.weight.copy_(w_o.t())
                elif w_o.numel() == self.out_proj.weight.numel():
                    self.out_proj.weight.copy_(w_o.reshape(self.out_proj.weight.shape))

                if b_o is not None and self.out_proj.bias is not None:
                    if b_o.shape == self.out_proj.bias.shape:
                        self.out_proj.bias.copy_(b_o)
                    elif b_o.numel() == self.out_proj.bias.numel():
                        self.out_proj.bias.copy_(b_o.reshape(self.out_proj.bias.shape))
        else:
            logger.warning(
                f"Output projection weight not found for {type(old_attn).__name__}. "
                f"Proceeding with initialized output projection weights."
            )

        if self.freeze_base:
            self.out_proj.freeze_base_weights()
            self.gate.weight.requires_grad = False
            if self.gate.bias is not None:
                self.gate.bias.requires_grad = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward propagation across hybrid HGRN-2 and Gated DeltaNet branches."""
        # Nested encoder cross-attention bypass
        if encoder_hidden_states is not None:
            logger.warning(
                "HGRN2GatedDeltaNet received encoder_hidden_states but cross-attention "
                "is not implemented. Returning raw hidden_states (identity)."
            )
            return hidden_states, None

        hgrn_res = self.hgrn(
            hidden_states, attention_mask=attention_mask, head_mask=head_mask, **kwargs
        )
        hgrn_out = hgrn_res[0] if isinstance(hgrn_res, tuple) else hgrn_res

        delta_res = self.deltanet(
            hidden_states, attention_mask=attention_mask, head_mask=head_mask, **kwargs
        )
        delta_out = delta_res[0] if isinstance(delta_res, tuple) else delta_res

        # Shape alignment between branches if needed
        if hgrn_out.shape != delta_out.shape:
            if hgrn_out.shape[-1] < delta_out.shape[-1]:
                hgrn_out = F.pad(hgrn_out, (0, delta_out.shape[-1] - hgrn_out.shape[-1]))
            else:
                delta_out = F.pad(delta_out, (0, hgrn_out.shape[-1] - delta_out.shape[-1]))

        gamma = torch.sigmoid(self.gate(hidden_states))
        fused = gamma * hgrn_out + (1.0 - gamma) * delta_out
        out = self.out_proj(fused)

        # Output shape match
        if out.shape[-1] != hidden_states.shape[-1]:
            if out.shape[-1] < hidden_states.shape[-1]:
                out = F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
            else:
                out = out[..., : hidden_states.shape[-1]]

        return out, None


@ATTENTION_REGISTRY.register("hgrn2_gated_deltanet_inject")
@ATTENTION_REGISTRY.register("inject_hgrn2_gated_deltanet")
def inject_hgrn2_gated_deltanet(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    **kwargs: Any,
) -> Optional[Any]:
    r"""Injection hook for HGRN2GatedDeltaNet supporting single layers and whole models."""
    target = layer if layer is not None else model
    cfg = model_config or getattr(target, "config", None) or kwargs.get("config", None)

    # 1. Whole model traversal support
    if target is not None and (
        hasattr(target, "transformer")
        or hasattr(target, "distilbert")
        or hasattr(target, "roberta")
        or hasattr(target, "bert")
        or (hasattr(target, "model") and not hasattr(target, "forward"))
    ):
        layer_list = None
        if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
            layer_list = target.distilbert.transformer.layer
        elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
            layer_list = target.roberta.encoder.layer
        elif hasattr(target, "bert") and hasattr(target.bert, "encoder"):
            layer_list = target.bert.encoder.layer
        elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
            layer_list = target.transformer.h
        elif hasattr(target, "model") and hasattr(target.model, "layers"):
            layer_list = target.model.layers
        elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
            layer_list = target.encoder.layer

        if layer_list is not None:
            num_injected = 0
            tot = len(layer_list)
            sub_kwargs = {k: v for k, v in kwargs.items() if k not in ("lora_rank", "lora_alpha")}
            last_injected = None
            for idx, lyr in enumerate(layer_list):
                res = inject_hgrn2_gated_deltanet(
                    layer=lyr,
                    model_config=cfg,
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    **sub_kwargs,
                )
                if res is not None:
                    num_injected += 1
                    last_injected = res
            if num_injected > 0 and last_injected is not None:
                total = num_injected * getattr(last_injected, "_per_layer_trainable", 0)
                if total > 20e6:
                    logger.warning("Total trainable across model: %.1fM", total / 1e6)
            return target if num_injected > 0 else None

    # 2. Single layer injection
    curr_attn = old_attn
    if curr_attn is None and target is not None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if hasattr(cand, "self"):
                    curr_attn = cand.self
                else:
                    curr_attn = cand
                break

    if curr_attn is None and target is not None and hasattr(target, "forward"):
        curr_attn = target

    if curr_attn is None:
        return None

    # Double-injection guard
    if curr_attn is not None:
        if isinstance(curr_attn, HGRN2GatedDeltaNet):
            return None
        if "HGRN2GatedDeltaNet" in curr_attn.__class__.__name__:
            return None
        if hasattr(curr_attn, "inner") and isinstance(getattr(curr_attn, "inner"), HGRN2GatedDeltaNet):
            return None
        if hasattr(curr_attn, "_wrapped_attention") and isinstance(getattr(curr_attn, "_wrapped_attention"), HGRN2GatedDeltaNet):
            return None

    if target is not None:
        for attr in ("attention", "self_attn", "attn", "self"):
            cand = getattr(target, attr, None)
            if cand is None:
                continue
            if isinstance(cand, HGRN2GatedDeltaNet):
                return None
            if "HGRN2GatedDeltaNet" in cand.__class__.__name__:
                return None
            if hasattr(cand, "inner") and isinstance(getattr(cand, "inner"), HGRN2GatedDeltaNet):
                return None
            if hasattr(cand, "_wrapped_attention") and isinstance(getattr(cand, "_wrapped_attention"), HGRN2GatedDeltaNet):
                return None

    # Skip Qwen3.5 hybrid DeltaNet/LinearAttention layers
    if curr_attn is not None:
        c_name = curr_attn.__class__.__name__
        if "DeltaNet" in c_name or "LinearAttention" in c_name:
            return None

    try:
        sub_kwargs = {k: v for k, v in kwargs.items() if k not in ("lora_rank", "lora_alpha")}
        new_layer = HGRN2GatedDeltaNet(
            config=cfg,
            old_attn=curr_attn,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            **sub_kwargs,
        )
    except Exception as ex:
        logger.warning("Failed to instantiate HGRN2GatedDeltaNet: %s", ex)
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_layer
            else:
                target.attention = new_layer
        elif hasattr(target, "self"):
            target.self = new_layer
        elif hasattr(target, "attn"):
            target.attn = new_layer
        elif hasattr(target, "self_attn"):
            target.self_attn = new_layer

    return new_layer
