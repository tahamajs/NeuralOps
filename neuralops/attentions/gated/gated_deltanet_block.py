"""
Gated DeltaNet Block with Optional Multi-Head Attention Bypass.

Implements a hybrid linear-recurrent block that dynamically switches between:
  * Standard multi-head self-attention (Softmax MHA)
  * Bidirectional Gated DeltaNet (linear recurrent, O(L) complexity)

Mathematical Formulation:
    Output = DeltaNet(X)  or  MHA(X, X, X)

The outer transformer block is responsible for the residual + LayerNorm:
    x_out = x + Block(x)
so this module MUST NOT add its own residual or normalization.

"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.attentions.gated_delta_net import GatedDeltaNet
from neuralops.core.adapters import UniversalAdapter
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# GatedDeltaNetBlock
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "gated_deltanet_block",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class GatedDeltaNetBlock(BaseAttention):
    r"""
    Gated DeltaNet Transformer Block with Pretrained Weight Transfer.

    Mathematical Formulation:
        Output = DeltaNet(X)  or  MHA(X, X, X)

    The outer transformer block handles the residual + normalization.
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
        use_full_attention: bool = False,
        comp_config: Any | None = None,
        layer_idx: int | None = None,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)

        # --- Resolve d_model ---
        if d_model is None:
            if config is not None:
                d_model = getattr(config, "hidden_size", getattr(config, "dim", 768))
            else:
                d_model = kwargs.get("hidden_size", 768)
        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        # --- Resolve num_heads ---
        if num_heads is None:
            if config is not None:
                num_heads = getattr(config, "num_attention_heads", 4)
            else:
                num_heads = kwargs.get("num_attention_heads", 4)

        self.d_model: int = int(d_model)
        self.num_heads: int = int(num_heads)
        self.use_full_attention: bool = bool(use_full_attention)
        self.comp_config = comp_config
        self.layer_idx = int(
            layer_idx if layer_idx is not None else kwargs.get("layer_idx", 0)
        )

        # --- Optional full-attention branch ---
        if self.use_full_attention:
            self.attn: nn.MultiheadAttention | None = nn.MultiheadAttention(
                self.d_model, self.num_heads, batch_first=True
            )
        else:
            self.attn = None

        # --- Gated DeltaNet branch (always constructed) ---
        source_attn = (
            pretrained_attn if pretrained_attn is not None else old_attn
        )
        self.deltanet = GatedDeltaNet(
            config=config,
            comp_config=comp_config,
            d_model=self.d_model,
            num_heads=self.num_heads,
            layer_idx=self.layer_idx,
            pretrained_attn=source_attn,
            **kwargs,
        )

        # --- Nested encoder detection ---
        self.is_nested_encoder: bool = bool(
            getattr(self.deltanet, "is_nested_encoder", False)
        )

        # --- Weight transfer if pretrained attention provided ---
        if source_attn is not None:
            self.copy_weights(source_attn)

    # ------------------------------------------------------------------
    # Compatibility properties (for HF integration)
    # ------------------------------------------------------------------
    @property
    def query(self) -> Any:
        return self.deltanet.query

    @property
    def key(self) -> Any:
        return self.deltanet.key

    @property
    def value(self) -> Any:
        return self.deltanet.value

    @property
    def q_proj(self) -> Any:
        return self.deltanet.q_proj

    @property
    def k_proj(self) -> Any:
        return self.deltanet.k_proj

    @property
    def v_proj(self) -> Any:
        return self.deltanet.v_proj

    @property
    def out_proj(self) -> Any:
        return self.deltanet.out_proj

    @property
    def q_lin(self) -> Any:
        return getattr(self.deltanet, "q_lin", self.deltanet.q_proj)

    @property
    def k_lin(self) -> Any:
        return getattr(self.deltanet, "k_lin", self.deltanet.k_proj)

    @property
    def v_lin(self) -> Any:
        return getattr(self.deltanet, "v_lin", self.deltanet.v_proj)

    @property
    def out_lin(self) -> Any:
        return getattr(self.deltanet, "out_lin", self.deltanet.out_proj)

    @property
    def dense(self) -> Any:
        return getattr(self.deltanet, "dense", self.deltanet.out_proj)

    @property
    def o_proj(self) -> Any:
        return getattr(self.deltanet, "o_proj", self.deltanet.out_proj)

    @property
    def c_proj(self) -> Any:
        return getattr(self.deltanet, "c_proj", self.deltanet.out_proj)

    # ------------------------------------------------------------------
    # copy_weights — GQA-aware
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: Any) -> None:
        r"""
        Transfer pretrained weights to either MHA or GatedDeltaNet.

        FIX: GQA-aware — handles the case where num_kv_heads < num_heads.
        For nn.MultiheadAttention, PyTorch requires all three (Q, K, V)
        to have the same head count. For GQA models (LLaMA, Qwen), we
        repeat/expand K and V to match Q's head count before concatenation.
        """
        if old_attn is None:
            return

        if self.use_full_attention and self.attn is not None:
            try:
                adapter = UniversalAdapter()
                w_q, b_q = adapter.get_weight_bias(old_attn, "q")
                w_k, b_k = adapter.get_weight_bias(old_attn, "k")
                w_v, b_v = adapter.get_weight_bias(old_attn, "v")
                w_o, b_o = adapter.get_weight_bias(old_attn, "o")

                # --- Fallback for output projection ---
                if w_o is None:
                    for cand in ("out_proj", "out_lin", "o_proj", "output", "proj"):
                        for target in (old_attn, getattr(old_attn, "self", None)):
                            if target is not None and hasattr(target, cand):
                                mod = getattr(target, cand)
                                if hasattr(mod, "dense"):
                                    mod = mod.dense
                                if hasattr(mod, "weight") and mod.weight is not None:
                                    w_o = mod.weight.data.clone()
                                    if w_o.ndim == 2 and w_o.shape[0] < w_o.shape[1]:
                                        w_o = w_o.t().contiguous()
                                    b = getattr(mod, "bias", None)
                                    b_o = b.data.clone() if b is not None else None
                                    break
                        if w_o is not None:
                            break

                if w_q is None or w_k is None or w_v is None:
                    logger.warning(
                        "GatedDeltaNetBlock: cannot extract Q/K/V for MHA transfer "
                        "from %s. Falling back to DeltaNet path.",
                        old_attn.__class__.__name__,
                    )
                    if hasattr(self.deltanet, "copy_weights"):
                        self.deltanet.copy_weights(old_attn)
                    return

                # --- GQA-AWARE EXPANSION ---
                # If K/V have fewer rows than Q (GQA), repeat them so their
                # row count matches Q. nn.MultiheadAttention expects equal dims.
                q_rows = w_q.shape[0]
                k_rows = w_k.shape[0]
                v_rows = w_v.shape[0]

                if k_rows != q_rows:
                    if q_rows % k_rows == 0:
                        rep = q_rows // k_rows
                        w_k = w_k.repeat(rep, 1)
                        if b_k is not None:
                            b_k = b_k.repeat(rep)
                        w_v = w_v.repeat(rep, 1)
                        if b_v is not None:
                            b_v = b_v.repeat(rep)
                        logger.info(
                            "GatedDeltaNetBlock: expanded GQA K/V by %d× to match Q heads.",
                            rep,
                        )
                    else:
                        logger.warning(
                            "GatedDeltaNetBlock: cannot match Q rows (%d) "
                            "to K rows (%d) for MHA transfer.",
                            q_rows,
                            k_rows,
                        )
                        if hasattr(self.deltanet, "copy_weights"):
                            self.deltanet.copy_weights(old_attn)
                        return

                with torch.no_grad():
                    # --- Fused QKV ---
                    qkv_w = torch.cat([w_q, w_k, w_v], dim=0)
                    if qkv_w.shape == self.attn.in_proj_weight.shape:
                        self.attn.in_proj_weight.copy_(
                            qkv_w.to(
                                device=self.attn.in_proj_weight.device,
                                dtype=self.attn.in_proj_weight.dtype,
                            )
                        )
                    else:
                        # Partial copy if shapes are close but not exact
                        out_d = min(qkv_w.size(0), self.attn.in_proj_weight.size(0))
                        in_d = min(qkv_w.size(1), self.attn.in_proj_weight.size(1))
                        self.attn.in_proj_weight.data[:out_d, :in_d].copy_(
                            qkv_w[:out_d, :in_d].to(self.attn.in_proj_weight.dtype)
                        )

                    # --- Fused QKV bias ---
                    if (
                        b_q is not None
                        and b_k is not None
                        and b_v is not None
                        and self.attn.in_proj_bias is not None
                    ):
                        cat_b = torch.cat([b_q, b_k, b_v], dim=0)
                        if cat_b.shape == self.attn.in_proj_bias.shape:
                            self.attn.in_proj_bias.copy_(
                                cat_b.to(self.attn.in_proj_bias.dtype)
                            )
                        else:
                            n = min(cat_b.numel(), self.attn.in_proj_bias.numel())
                            self.attn.in_proj_bias.data[:n].copy_(
                                cat_b[:n].to(self.attn.in_proj_bias.dtype)
                            )

                    # --- Output projection ---
                    if w_o is not None:
                        if w_o.shape == self.attn.out_proj.weight.shape:
                            self.attn.out_proj.weight.copy_(
                                w_o.to(self.attn.out_proj.weight.dtype)
                            )
                        elif w_o.t().contiguous().shape == self.attn.out_proj.weight.shape:
                            self.attn.out_proj.weight.copy_(
                                w_o.t().contiguous().to(self.attn.out_proj.weight.dtype)
                            )
                        else:
                            out_d = min(w_o.size(0), self.attn.out_proj.weight.size(0))
                            in_d = min(w_o.size(1), self.attn.out_proj.weight.size(1))
                            self.attn.out_proj.weight.data[:out_d, :in_d].copy_(
                                w_o[:out_d, :in_d].to(self.attn.out_proj.weight.dtype)
                            )

                        if b_o is not None and self.attn.out_proj.bias is not None:
                            n = min(b_o.numel(), self.attn.out_proj.bias.numel())
                            self.attn.out_proj.bias.data[:n].copy_(
                                b_o[:n].to(self.attn.out_proj.bias.dtype)
                            )

                    # --- Frobenius norm verification ---
                    for name, w in (
                        ("qkv", self.attn.in_proj_weight),
                        ("o", self.attn.out_proj.weight),
                    ):
                        w_norm = float(torch.linalg.norm(w.data.float()).item())
                        if w_norm < 1e-4:
                            logger.warning(
                                "GatedDeltaNetBlock: '%s' weight has near-zero "
                                "Frobenius norm (%.2e).",
                                name,
                                w_norm,
                            )
            except Exception as e:
                logger.warning(
                    "GatedDeltaNetBlock copy_weights to MultiheadAttention incomplete: %s",
                    e,
                )
                if hasattr(self.deltanet, "copy_weights"):
                    self.deltanet.copy_weights(old_attn)
        else:
            if hasattr(self.deltanet, "copy_weights"):
                self.deltanet.copy_weights(old_attn)
                self.is_nested_encoder = bool(
                    getattr(self.deltanet, "is_nested_encoder", False)
                )

    # ------------------------------------------------------------------
    # Forward — no internal residual / norm
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
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        """
        Forward pass.

        IMPORTANT: This module is a drop-in replacement for the attention sublayer
        of a transformer block. It MUST NOT add its own residual or LayerNorm,
        because the outer block already does:
            x_out = x + Block(x)
        """
        x = hidden_states

        if self.use_full_attention and self.attn is not None:
            # --- MHA branch ---
            key_padding_mask = None
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    key_padding_mask = attention_mask == 0
                elif attention_mask.dim() == 4:
                    key_padding_mask = attention_mask[:, 0, 0, :] <= -10000.0

            # NOTE: nn.MultiheadAttention has no RoPE. For decoder models with
            # use_full_attention=True, positional information is lost. Recommend
            # using the DeltaNet path for decoders.

            attn_out, attn_weights = self.attn(
                x,
                x,
                x,
                key_padding_mask=key_padding_mask,
                need_weights=bool(output_attentions),
            )

            if attn_out.shape[-1] != x.shape[-1]:
                attn_out = attn_out[..., : x.shape[-1]]

            # No internal residual, no internal norm.
            return attn_out, (attn_weights if output_attentions else None)

        else:
            # --- Gated DeltaNet branch ---
            # Forward ALL relevant kwargs so KV-cache and RoPE work.
            delta_result = self.deltanet(
                x,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                *args,
                **kwargs,
            )

            # FIX: extract both output AND present state / cache
            if isinstance(delta_result, tuple):
                delta_out = delta_result[0]
                present_state = delta_result[1] if len(delta_result) > 1 else None
            else:
                delta_out = delta_result
                present_state = None

            if delta_out.shape[-1] != x.shape[-1]:
                delta_out = delta_out[..., : x.shape[-1]]

            delta_out = torch.nan_to_num(delta_out, nan=0.0, posinf=1e4, neginf=-1e4)
            # No internal residual, no internal norm.
            return delta_out, present_state

    # ------------------------------------------------------------------
    # Orthogonality loss hook (no-op)
    # ------------------------------------------------------------------
    def get_ortho_loss(self) -> torch.Tensor:
        try:
            return torch.tensor(0.0, device=self.deltanet.out_proj.weight.device)
        except Exception:
            return torch.tensor(0.0)


# ===========================================================================
# Injection
# ===========================================================================
def inject_gated_deltanet_block(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_gated_deltanet_block: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = (
        model_config
        if model_config is not None
        else getattr(target, "config", None)
    )

    # --- Whole-model traversal ---
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (torch.nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_gated_deltanet_block(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # --- Single-layer injection ---
    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "attn", "self_attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, torch.nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break
        if src_attn is None:
            src_attn = target

    # --- Idempotency guard ---
    if src_attn is not None and (
        isinstance(src_attn, GatedDeltaNetBlock)
        or "GatedDeltaNetBlock" in src_attn.__class__.__name__
    ):
        return None

    # --- Resolve d_model / num_heads ---
    d_model = 64
    num_heads = 4
    if src_attn is not None and src_attn is not target:
        cand = getattr(
            src_attn,
            "hidden_size",
            getattr(src_attn, "embed_dim", getattr(src_attn, "all_head_size", None)),
        )
        if cand is None and hasattr(src_attn, "q_proj") and hasattr(
            src_attn.q_proj, "in_features"
        ):
            cand = src_attn.q_proj.in_features
        if cand is not None:
            d_model = cand
        num_heads = getattr(
            src_attn, "num_heads", getattr(src_attn, "num_attention_heads", num_heads)
        )

    if cfg is not None:
        d_model = getattr(cfg, "hidden_size", getattr(cfg, "dim", d_model))
        num_heads = getattr(cfg, "num_attention_heads", num_heads)

    try:
        use_full = getattr(comp_config, "use_full_attention", False) if comp_config else False
        new_block = GatedDeltaNetBlock(
            config=cfg,
            comp_config=comp_config,
            d_model=d_model,
            num_heads=num_heads,
            use_full_attention=use_full,
            layer_idx=layer_idx,
            pretrained_attn=src_attn if src_attn is not target else None,
            **kwargs,
        )

        if src_attn is not None and src_attn is not target:
            new_block.copy_weights(src_attn)

        # --- Move to same device/dtype as source ---
        if src_attn is not None and src_attn is not target:
            p = next(src_attn.parameters(), None)
            if p is not None:
                new_block.to(device=p.device, dtype=p.dtype)

        # --- Splice ---
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_block.is_nested_encoder = True
                new_block.deltanet.is_nested_encoder = True
                target.attention.self = new_block
            else:
                new_block.is_nested_encoder = False
                new_block.deltanet.is_nested_encoder = False
                target.attention = new_block
        elif hasattr(target, "self"):
            new_block.is_nested_encoder = True
            new_block.deltanet.is_nested_encoder = True
            target.self = new_block
        elif hasattr(target, "attn"):
            new_block.is_nested_encoder = False
            new_block.deltanet.is_nested_encoder = False
            target.attn = new_block
        elif hasattr(target, "self_attn"):
            new_block.is_nested_encoder = False
            new_block.deltanet.is_nested_encoder = False
            target.self_attn = new_block
        else:
            logger.error(
                "inject_gated_deltanet_block: no writable attention attribute "
                "on %s",
                target.__class__.__name__,
            )
            return None

        # --- Sanity check ---
        assert (
            new_block.is_nested_encoder == new_block.deltanet.is_nested_encoder
        ), "GatedDeltaNetBlock nested flag desynced"

        return new_block

    except Exception as e:
        logger.exception("GatedDeltaNetBlock injection failed: %s", e)
        return None


__all__ = ["GatedDeltaNetBlock", "inject_gated_deltanet_block"]