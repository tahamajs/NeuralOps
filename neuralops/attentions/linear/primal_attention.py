"""
Primal Attention: Self-Attention through Asymmetric Kernel SVD in Primal Representation.

Nyström-inspired low-rank landmark approximation of self-attention using segment pooling
landmarks and iterative Schulz pseudoinverse refinement.
"""

from __future__ import annotations

from neuralops.core.attention_utils import (
    apply_rotary_pos_emb,
    LoRALinear,
)

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import apply_attention_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

logger = logging.getLogger(__name__)


# ===========================================================================
# Primal Attention
# ===========================================================================


@ATTENTION_REGISTRY.register("primal_attention")
class PrimalAttention(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int = 64,
        pinv_iters: int = 2,
        pretrained_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        if rank is None:
            rank = 64
        super().__init__(config, rank)
        self.config = config
        comp_config = getattr(config, "compression_config", None)

        self.dim = getattr(config, "hidden_size", kwargs.get("hidden_size", 768))
        self.num_heads = getattr(
            config, "num_attention_heads", kwargs.get("num_attention_heads", 12)
        )
        self.num_kv_heads = getattr(
            config,
            "num_key_value_heads",
            kwargs.get("num_key_value_heads", self.num_heads),
        )
        self.head_dim = getattr(
            config,
            "head_dim",
            kwargs.get("head_dim", self.dim // self.num_heads),
        )
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim

        # PR-FIX-03: guard self.rank against None
        _rank_candidate = None
        if comp_config is not None:
            _rank_candidate = getattr(comp_config, "rank", None)
        if _rank_candidate is None:
            _rank_candidate = kwargs.pop("rank", None)
        if _rank_candidate is None:
            _rank_candidate = rank
        self.rank = int(_rank_candidate) if _rank_candidate else 64

        self.pinv_iters = pinv_iters

        is_dec = getattr(config, "is_decoder", kwargs.get("is_decoder", False))
        if kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        self.lora_rank = (
            getattr(
                config,
                "lora_rank",
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
            )
            if config
            else 16
        )
        self.lora_alpha = (
            getattr(
                config,
                "lora_alpha",
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
            )
            if config
            else 32.0
        )
        self.freeze_base = self.lora_rank > 0
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        # PR-FIX-07: chunked head processing threshold.
        # If rank is large, split heads into chunks of CHUNK_HEADS to bound memory.
        self.chunk_heads = int(kwargs.get("chunk_heads", 0))  # 0 = disabled

        # Projections
        self.query = LoRALinear(
            self.dim,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.key = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.value = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        self.q_proj = self.query
        self.k_proj = self.key
        self.v_proj = self.value

        dropout_p = (
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        source_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("source_attn")
        if source_attn is not None:
            self.copy_weights(source_attn)

    # ------------------------------------------------------------------
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if (
            (
                hasattr(src, "self")
                and hasattr(src.self, "query")
                and hasattr(src, "output")
                and hasattr(src.output, "dense")
            )
            or (
                cls_name
                in [
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                    "ElectraSelfAttention",
                    "DebertaSelfAttention",
                ]
                or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
            )
            and not has_out_lin
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2 fused
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                is_conv = (
                    c_attn_mod.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                )
                if is_conv:
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (
                    c_w[:split_size, :],
                    c_b[:split_size] if c_b is not None else None,
                )
                extracted["k"] = (
                    c_w[split_size : 2 * split_size, :],
                    c_b[split_size : 2 * split_size] if c_b is not None else None,
                )
                extracted["v"] = (
                    c_w[2 * split_size :, :],
                    c_b[2 * split_size :] if c_b is not None else None,
                )

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    is_conv_o = (
                        c_proj_mod.__class__.__name__ == "Conv1D"
                        or base_o.__class__.__name__ == "Conv1D"
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    )
                    if is_conv_o:
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (
                        o_w,
                        bias_o.data.clone() if bias_o is not None else None,
                    )

        # 2. RoBERTa / BERT nested
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (
                        mod.weight.data.clone(),
                        mod.bias.data.clone() if mod.bias is not None else None,
                    )
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                out_mod = src.output.dense
                extracted["o"] = (
                    out_mod.weight.data.clone(),
                    out_mod.bias.data.clone() if out_mod.bias is not None else None,
                )

        # 3. Flat modules (DistilBERT, GPT-2 style without c_attn)
        else:
            proj_candidates = {
                "q": ["q_lin", "q_proj", "query"],
                "k": ["k_lin", "k_proj", "key"],
                "v": ["v_lin", "v_proj", "value"],
                "o": ["out_lin", "out_proj", "dense", "o_proj"],
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(src, name):
                        mod = getattr(src, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj_key] = (
                                mod.weight.data.clone(),
                                mod.bias.data.clone() if mod.bias is not None else None,
                            )
                            break

        missing = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]

        if missing:
            try:
                from neuralops.core.adapters import UniversalAdapter

                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            raise RuntimeError(
                f"PrimalAttention copy_weights failed: missing mandatory projections "
                f"{missing_after} from {src.__class__.__name__}."
            )

        target_map = {
            "q": self.query,
            "k": self.key,
            "v": self.value,
            "o": self.out_proj,
        }

        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]

                if w is None:
                    if proj_key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target_mod.weight)
                        if target_mod.bias is not None:
                            nn.init.zeros_(target_mod.bias)
                        continue
                    else:
                        logger.warning(
                            "PrimalAttention.copy_weights: no weight for %s, "
                            "keeping initialized weights.",
                            proj_key,
                        )
                        continue

                # Shape alignment
                if w.shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w)
                elif w.t().contiguous().shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w.t().contiguous())
                elif (
                    w.shape[0] < target_mod.weight.shape[0]
                    and target_mod.weight.shape[0] % w.shape[0] == 0
                ):
                    repeats = target_mod.weight.shape[0] // w.shape[0]
                    target_mod.weight.copy_(w.repeat(repeats, 1))
                else:
                    min_0 = min(target_mod.weight.shape[0], w.shape[0])
                    min_1 = min(target_mod.weight.shape[1], w.shape[1])
                    target_mod.weight.zero_()
                    target_mod.weight[:min_0, :min_1].copy_(w[:min_0, :min_1])

                if b is not None and target_mod.bias is not None:
                    if b.shape == target_mod.bias.shape:
                        target_mod.bias.copy_(b)
                    elif (
                        b.shape[0] < target_mod.bias.shape[0]
                        and target_mod.bias.shape[0] % b.shape[0] == 0
                    ):
                        repeats = target_mod.bias.shape[0] // b.shape[0]
                        target_mod.bias.copy_(b.repeat(repeats))
                    else:
                        min_b = min(target_mod.bias.shape[0], b.shape[0])
                        target_mod.bias.zero_()
                        target_mod.bias[:min_b].copy_(b[:min_b])
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                fnorm = torch.linalg.norm(target_mod.weight.float()).item()
                if fnorm < 1e-4:
                    logger.warning(
                        "PrimalAttention.copy_weights: %s has near-zero Frobenius norm (%.2e).",
                        proj_key,
                        fnorm,
                    )

        if self.freeze_base:
            for mod in [self.query, self.key, self.value, self.out_proj]:
                mod.freeze_base_weights()

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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = hidden_states.shape

        Q = (
            self.query(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        K = (
            self.key(hidden_states)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        V = (
            self.value(hidden_states)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # RoPE
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # GQA expansion
        if self.num_heads != self.num_kv_heads:
            num_reps = self.num_heads // self.num_kv_heads
            K = repeat_kv(K, num_reps)
            V = repeat_kv(V, num_reps)

        # Landmark computation
        num_landmarks = min(self.rank, seq_len)
        if seq_len % num_landmarks == 0:
            seg = seq_len // num_landmarks
            Q_land = Q.reshape(batch_size, self.num_heads, num_landmarks, seg, self.head_dim).mean(
                dim=3
            )
            K_land = K.reshape(batch_size, self.num_heads, num_landmarks, seg, self.head_dim).mean(
                dim=3
            )
        else:
            idx = torch.linspace(
                0,
                seq_len - 1,
                num_landmarks,
                dtype=torch.long,
                device=hidden_states.device,
            )
            Q_land = Q[:, :, idx, :]
            K_land = K[:, :, idx, :]

        scale = 1.0 / (self.head_dim**0.5)
        is_causal = bool(self.is_decoder or kwargs.get("is_causal", False) or kwargs.get("causal", False))

        if is_causal:
            # Exact causal attention to prevent future token leakage across landmarks
            scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), -1e4, device=scores.device),
                diagonal=1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
            if attention_mask is not None:
                scores = apply_attention_mask(scores, attention_mask)
            probs = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0).to(hidden_states.dtype)
            probs = self.dropout(probs)
            if head_mask is not None:
                probs = probs * head_mask
            context = torch.matmul(probs, V)
            context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.all_head_size)
            if self.is_nested_encoder:
                return context, (probs if output_attentions else None)
            output = self.out_proj(context)
            return output, (probs if output_attentions else None)

        # PR-FIX-01 + PR-FIX-07: chunked head processing to bound peak memory.
        # The chunk_heads attribute (default 0) disables chunking.
        if self.chunk_heads > 0 and self.num_heads > self.chunk_heads:
            head_chunks = [
                (h0, min(h0 + self.chunk_heads, self.num_heads))
                for h0 in range(0, self.num_heads, self.chunk_heads)
            ]
        else:
            head_chunks = [(0, self.num_heads)]

        context_heads = []
        attn_for_return: torch.Tensor | None = None

        for h0, h1 in head_chunks:
            Qc = Q[:, h0:h1]
            Kc = K[:, h0:h1]
            Vc = V[:, h0:h1]
            Ql = Q_land[:, h0:h1]
            Kl = K_land[:, h0:h1]

            with torch.amp.autocast("cuda", enabled=False):
                F1 = torch.matmul(Qc.float(), Kl.float().transpose(-2, -1).contiguous()) * scale
                F2 = torch.matmul(Ql.float(), Kl.float().transpose(-2, -1).contiguous()) * scale
                F3 = torch.matmul(Ql.float(), Kc.float().transpose(-2, -1).contiguous()) * scale

                # PR-FIX-01: pseudoinverse + Schulz without gradient tracking.
                # This eliminates the ~20x backward memory of pinv and stabilizes
                # the iteration numerically (rcond clamps near-singular values).
                with torch.no_grad():
                    F2_f32 = F2.detach().to(torch.float32)
                    F2_pinv = torch.linalg.pinv(F2_f32, rcond=1e-4)
                    I = (
                        torch.eye(
                            num_landmarks,
                            device=hidden_states.device,
                            dtype=torch.float32,
                        )
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    X = F2_pinv
                    for _ in range(min(self.pinv_iters, 4)):
                        X = torch.matmul(X, 2.0 * I - torch.matmul(F2_f32, X))
                    X = X.detach()

                    # PR-FIX-02: free pinv intermediates.
                    del F2_pinv, I

                attn_c = torch.matmul(F1, torch.matmul(X, F3))

                del F1, F2, F3, F2_f32, X

            # Causal mask
            is_causal = self.is_decoder or kwargs.get("is_causal", False)
            if is_causal:
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), -1e4, device=attn_c.device),
                    diagonal=1,
                )
                attn_c = attn_c + causal_mask.unsqueeze(0).unsqueeze(0)

            # Padding mask
            if attention_mask is not None:
                attn_c = apply_attention_mask(attn_c, attention_mask)

            probs_c = F.softmax(attn_c, dim=-1).to(hidden_states.dtype)
            probs_c = torch.nan_to_num(probs_c, nan=0.0)
            probs_c = self.dropout(probs_c)

            # head_mask (per-head, so select chunk)
            if head_mask is not None:
                if head_mask.dim() == 1 and head_mask.size(0) == self.num_heads:
                    hm = head_mask[h0:h1].view(1, h1 - h0, 1, 1).to(probs_c.dtype)
                    probs_c = probs_c * hm
                elif head_mask.dim() == 2 and head_mask.size(1) == self.num_heads:
                    hm = (
                        head_mask[:, h0:h1].view(head_mask.size(0), h1 - h0, 1, 1).to(probs_c.dtype)
                    )
                    probs_c = probs_c * hm
                elif head_mask.dim() == 4:
                    hm = head_mask[:, h0:h1].to(probs_c.dtype)
                    probs_c = probs_c * hm

            ctx_c = torch.matmul(probs_c, Vc)
            context_heads.append(ctx_c)

            if output_attentions and attn_for_return is None:
                # PR-FIX-05: return softmax probs, not logits.
                attn_for_return = probs_c

        # Concatenate chunks along head axis: (B, H_total, N, D)
        context = torch.cat(context_heads, dim=1)
        context = (
            context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.all_head_size)
        )

        if self.is_nested_encoder:
            return context, (attn_for_return if output_attentions else None)

        out = self.out_proj(context)
        return out, (attn_for_return if output_attentions else None)


# ===========================================================================
# Injection
# ===========================================================================


def inject_primal(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    target_layer_idx: int | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    """PR-FIX-06: added recursion guard and removed the dead `in locals()` check."""
    if _depth > 8:
        logger.warning("inject_primal: max recursion depth exceeded, aborting.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (nn.ModuleList, list)):
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
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_primal(
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

    layer = target

    try:
        if layer is not None:
            config = model_config or getattr(layer, "config", None)
            src = old_attn
            if src is None:
                for attr in ("attention", "self_attn", "attn", "self"):
                    if hasattr(layer, attr):
                        cand = getattr(layer, attr)
                        if isinstance(cand, nn.Module) and cand is not layer:
                            src = cand
                            break
            if src is None:
                src = layer

            new_attn = PrimalAttention(
                config=config,
                pretrained_attn=src if src is not layer else None,
                **kwargs,
            )

            # Attach in-place
            for attr in ("self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    setattr(layer, attr, new_attn)
                    return layer
            if hasattr(layer, "attention"):
                if hasattr(layer.attention, "self"):
                    layer.attention.self = new_attn
                else:
                    layer.attention = new_attn
                return layer
            layer.attention = new_attn
            return layer

        if model is None:
            return None

        replaced = 0
        for name, module in model.named_modules():
            is_attn = (
                hasattr(module, "q_lin")
                or hasattr(module, "query")
                or hasattr(module, "q_proj")
                or hasattr(module, "c_attn")
            )
            if is_attn:
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                child_name = name.rsplit(".", 1)[-1] if "." in name else name
                parent = model.get_submodule(parent_name) if parent_name else model

                new_attn = PrimalAttention(
                    config=getattr(model, "config", None),
                    pretrained_attn=module,
                    **kwargs,
                )
                setattr(parent, child_name, new_attn)
                replaced += 1
                if target_layer_idx is not None and replaced > target_layer_idx:
                    break
        return model if replaced > 0 else None
    except Exception as e:
        logger.exception("inject_primal failed: %s", e)
        return None


inject_primal_attention = inject_primal
