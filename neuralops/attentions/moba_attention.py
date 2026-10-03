"""MoBA: Mixture of Block Attention — Production-Hardened (v2).

Registered as 'moba', 'moba_attention', 'moba_inject'.

"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE
# ===========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None:
        return q, k

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    if cos.shape[-2] != q.shape[-2]:
        cos = cos[..., : q.shape[-2], :]
        sin = sin[..., : q.shape[-2], :]

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
        k_rot_embed = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
        q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
        k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))

    return q, k


# ===========================================================================
# LoRALinear
# ===========================================================================
class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        lora_rank=16,
        lora_alpha=32.0,
        freeze_base=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **fk))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **fk))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **fk))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **fk))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

    def freeze_base_weights(self):
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def forward(self, x):
        # MB-FIX-12: dtype-safe
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = (
            self.bias.to(x.dtype)
            if (self.bias is not None and self.bias.dtype != x.dtype)
            else self.bias
        )
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            out = out + F.linear(F.linear(x, a), bb) * self.lora_scaling
        return out


# ===========================================================================
# MoBA
# ===========================================================================
@ATTENTION_REGISTRY.register("moba", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("moba_attention", supports_causal=True, supports_gqa=True)
class MoBAAttention(BaseAttention):
    """
    MoBA: Mixture of Block Attention for Long-Context Modeling.

    For each query block, we compute block-level centroid scores, select the
    top-k most relevant key blocks, and run dense attention within the
    selected blocks (plus the causal-visible diagonal block).
    """

    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True
    triton_kernel: str = "sparse_attn"

    def __init__(
        self,
        config: Any,
        block_size: int = 64,
        top_k: int = 4,
        rank: int | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        # MB-FIX-03
        super().__init__(config, rank if rank is not None else 32)

        # MB-FIX-04: config None-safe
        if config is None:

            class _Cfg:
                pass

            config = _Cfg()
            config.hidden_size = kwargs.get("hidden_size", 768)
            config.num_attention_heads = kwargs.get("num_attention_heads", 12)
            config.head_dim = kwargs.get("head_dim", 64)
            config.model_type = kwargs.get("model_type", "")

        self.num_heads = int(
            getattr(
                config,
                "num_attention_heads",
                getattr(config, "num_heads", kwargs.get("num_attention_heads", 4)),
            )
        )
        self.hidden_size = int(getattr(config, "hidden_size", kwargs.get("hidden_size", 768)))
        cfg_hd = getattr(config, "head_dim", None)
        self.head_dim = (
            int(cfg_hd)
            if cfg_hd is not None
            else max(1, self.hidden_size // max(1, self.num_heads))
        )
        self.all_head_size = self.num_heads * self.head_dim

        cfg_kv = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None))
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat = self.num_heads // self.num_kv_heads
        self.kv_head_size = self.num_kv_heads * self.head_dim

        self.block_size = int(block_size)
        self.top_k = int(top_k)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)

        # ---- decoder detection ----
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            _mt = str(getattr(config, "model_type", "") or "").lower()
            self.is_decoder = bool(
                getattr(config, "is_decoder", False) or getattr(config, "causal", False)
            ) or any(
                _mt.startswith(p)
                for p in (
                    "gpt",
                    "qwen",
                    "llama",
                    "mistral",
                    "falcon",
                    "bloom",
                    "opt",
                    "gemma",
                    "phi",
                    "olmo",
                    "mpt",
                    "stablelm",
                )
            )

        self.is_nested_encoder: bool = False

        # ---- MB-FIX-01: output scale (initialized to 1.0) ----
        self.output_scale = nn.Parameter(torch.ones(1))

        dropout_p = float(
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
        )
        self.dropout = nn.Dropout(dropout_p)

        # ---- projections ----
        self.q_proj = LoRALinear(
            self.hidden_size,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.k_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.v_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        # MB-FIX-11: storage accounting
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0
        self.triton_kwargs = {"block_size": self.block_size, "eps": 1e-6}

        pretrained_attn = kwargs.get("pretrained_attn")
        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ---- MB-FIX-07: property aliases ----
    @property
    def query(self):
        return self.q_proj

    @property
    def key(self):
        return self.k_proj

    @property
    def value(self):
        return self.v_proj

    @property
    def q_lin(self):
        return self.q_proj

    @property
    def k_lin(self):
        return self.k_proj

    @property
    def v_lin(self):
        return self.v_proj

    @property
    def out_lin(self):
        return self.out_proj

    @property
    def dense(self):
        return self.out_proj

    @property
    def c_proj(self):
        return self.out_proj

    @property
    def o_proj(self):
        return self.out_proj

    # ------------------------------------------------------------------
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        from neuralops.core.adapters import UniversalAdapter

        # Track original storage
        try:
            for p in src.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. UniversalAdapter
        for proj in ("q", "k", "v", "o"):
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        # 2. GPT-2 fused c_attn
        if hasattr(src, "c_attn") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            c_attn = src.c_attn
            base_c = getattr(c_attn, "base_layer", c_attn)
            cw = getattr(base_c, "weight", getattr(c_attn, "weight", None))
            if cw is not None:
                cw = cw.data.clone()
                if cw.dim() == 2 and cw.shape[0] < cw.shape[1]:
                    cw = cw.t().contiguous()
                cb = getattr(base_c, "bias", getattr(c_attn, "bias", None))
                cb_data = cb.data.clone() if cb is not None else None
                split = cw.shape[0] // 3
                extracted["q"] = (
                    cw[:split],
                    cb_data[:split] if cb_data is not None else None,
                )
                extracted["k"] = (
                    cw[split : 2 * split],
                    cb_data[split : 2 * split] if cb_data is not None else None,
                )
                extracted["v"] = (
                    cw[2 * split :],
                    cb_data[2 * split :] if cb_data is not None else None,
                )

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            cp = src.c_proj
            base_o = getattr(cp, "base_layer", cp)
            ow = getattr(base_o, "weight", getattr(cp, "weight", None))
            if ow is not None:
                ow = ow.data.clone()
                if ow.dim() == 2 and ow.shape[0] < ow.shape[1]:
                    ow = ow.t().contiguous()
                bo = getattr(base_o, "bias", getattr(cp, "bias", None))
                extracted["o"] = (ow, bo.data.clone() if bo is not None else None)

        # 3. RoBERTa / BERT nested (MB-FIX-02: pass .data, not module)
        if hasattr(src, "self") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            s = src.self
            if hasattr(s, "query") and extracted["q"][0] is None:
                q_l = getattr(s.query, "base_layer", s.query)
                extracted["q"] = (
                    q_l.weight.data,
                    q_l.bias.data if getattr(q_l, "bias", None) is not None else None,
                )
            if hasattr(s, "key") and extracted["k"][0] is None:
                k_l = getattr(s.key, "base_layer", s.key)
                extracted["k"] = (
                    k_l.weight.data,
                    k_l.bias.data if getattr(k_l, "bias", None) is not None else None,
                )
            if hasattr(s, "value") and extracted["v"][0] is None:
                v_l = getattr(s.value, "base_layer", s.value)
                extracted["v"] = (
                    v_l.weight.data,
                    v_l.bias.data if getattr(v_l, "bias", None) is not None else None,
                )

        if hasattr(src, "output") and hasattr(src.output, "dense") and extracted["o"][0] is None:
            d = getattr(src.output.dense, "base_layer", src.output.dense)
            extracted["o"] = (
                d.weight.data,
                d.bias.data if getattr(d, "bias", None) is not None else None,
            )

        # 4. Modern decoders / DistilBERT
        named_map = {
            "q": ("q_proj", "q_lin", "query"),
            "k": ("k_proj", "k_lin", "key"),
            "v": ("v_proj", "v_lin", "value"),
            "o": ("o_proj", "out_proj", "out_lin", "dense"),
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(src, name):
                        m = getattr(src, name)
                        w = getattr(m, "weight", None)
                        if w is not None and hasattr(w, "data"):
                            b = getattr(m, "bias", None)
                            extracted[proj] = (
                                w.data,
                                b.data if b is not None else None,
                            )
                            break

        # MB-FIX-06: nested encoder detection after all extraction
        cls_name = src.__class__.__name__
        has_out_proj = any(
            hasattr(src, a)
            for a in (
                "dense",
                "out_proj",
                "c_proj",
                "out_lin",
                "o_proj",
                "wo",
                "o",
                "proj",
            )
        )
        is_nested = False
        if (
            hasattr(src, "self")
            and hasattr(src.self, "query")
            or cls_name in ("BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention")
            or cls_name.endswith("SelfAttention")
            and not has_out_proj
            or cls_name.endswith("Attention")
            and not has_out_proj
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning(
                "MoBA copy_weights: missing %s; proceeding with initialized weights.",
                missing,
            )

        # Frobenius norm checks
        for k_ in ("q", "k", "v"):
            w = extracted[k_][0]
            if w is not None:
                n = torch.linalg.norm(w.float()).item()
                if n < 1e-4:
                    logger.warning("MoBA: %s weight near-zero norm (%.2e)", k_, n)
        if extracted["o"][0] is not None:
            n = torch.linalg.norm(extracted["o"][0].float()).item()
            if n < 1e-4:
                logger.warning("MoBA: o weight near-zero norm (%.2e)", n)

        def _orient(w, in_dim):
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        # ---- Copy ----
        with torch.no_grad():
            for proj_key, target in (
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
                ("o", self.out_proj),
            ):
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                    w_oriented = _orient(w, in_dim)
                    target_w = target.weight
                    out_d, in_d = target_w.shape
                    src_out, src_in = w_oriented.shape

                    if proj_key in ("k", "v"):
                        if src_out == out_d:
                            target_w.data.copy_(w_oriented)
                            if b is not None and target.bias is not None:
                                target.bias.data.copy_(b)
                        elif src_out > out_d and src_out % out_d == 0:
                            rep = src_out // out_d
                            target_w.data.copy_(w_oriented.view(out_d, rep, -1).mean(dim=1))
                            if b is not None and target.bias is not None:
                                target.bias.data.copy_(b.view(out_d, rep).mean(dim=1))
                        elif src_out < out_d and out_d % src_out == 0:
                            rep = out_d // src_out
                            target_w.data.copy_(w_oriented.repeat(rep, 1))
                            if b is not None and target.bias is not None:
                                target.bias.data.copy_(b.repeat(rep))
                        else:
                            m0, m1 = min(out_d, src_out), min(in_d, src_in)
                            target_w.data[:m0, :m1].copy_(w_oriented[:m0, :m1])
                            if b is not None and target.bias is not None:
                                mb = min(target.bias.shape[0], b.shape[0])
                                target.bias.data[:mb].copy_(b[:mb])
                    else:
                        m0, m1 = min(out_d, src_out), min(in_d, src_in)
                        target_w.data[:m0, :m1].copy_(w_oriented[:m0, :m1])
                        if b is not None and target.bias is not None:
                            mb = min(target.bias.shape[0], b.shape[0])
                            target.bias.data[:mb].copy_(b[:mb])
                elif proj_key == "o" and self.is_nested_encoder:
                    min_d = min(target.weight.shape[0], target.weight.shape[1])
                    target.weight.data.zero_()
                    target.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                    if target.bias is not None:
                        target.bias.data.zero_()

                target.freeze_base_weights()

        # Track compressed storage
        try:
            for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        # ---- is_decoder override ----
        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)
        elif (
            cls_name in ("GPT2Attention", "GPT2Block")
            or hasattr(src, "c_attn")
            or "gpt" in cls_name.lower()
        ):
            self.is_decoder = True

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
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        L_kv = kv_input.shape[1]

        Q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2).contiguous()
        K = self.k_proj(kv_input).view(B, L_kv, self.num_kv_heads, d).transpose(1, 2).contiguous()
        V = self.v_proj(kv_input).view(B, L_kv, self.num_kv_heads, d).transpose(1, 2).contiguous()

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if self.num_kv_heads < self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        is_causal = bool(self.is_decoder and not is_cross and L > 1)
        scale = 1.0 / math.sqrt(d)

        eff_mask = encoder_attention_mask if is_cross else attention_mask

        # ---- Normalize mask to bool (B, L_kv) ----
        valid_mask: torch.Tensor | None = None
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                if eff_mask.dtype == torch.bool:
                    valid_mask = eff_mask
                elif eff_mask.dtype.is_floating_point:
                    valid_mask = eff_mask > -1e4
                else:
                    valid_mask = eff_mask > 0
            elif eff_mask.dim() == 3:
                m = eff_mask
                if m.dtype == torch.bool:
                    valid_mask = m.squeeze(1)
                elif m.dtype.is_floating_point:
                    valid_mask = (m > -1e4).squeeze(1)
                else:
                    valid_mask = (m > 0).squeeze(1)
            elif eff_mask.dim() == 4:
                m = eff_mask
                if m.dtype == torch.bool:
                    vv = m
                elif m.dtype.is_floating_point:
                    vv = m > -1e4
                else:
                    vv = m > 0
                valid_mask = vv.any(dim=-2).squeeze(1)

            if valid_mask is not None and valid_mask.shape[-1] != L_kv:
                valid_mask = None  # shape mismatch → ignore

        # ---- Fast path: short sequence or cross-attention ----
        if self.block_size >= L or is_cross:
            if valid_mask is None and head_mask is None and not output_attentions:
                try:
                    _qs = Q.half() if Q.dtype == torch.float32 else Q
                    _ks = K.half() if K.dtype == torch.float32 else K
                    _vs = V.half() if V.dtype == torch.float32 else V
                    _is_causal = self.is_decoder and not is_cross and L > 1
                    _sdpa_out = F.scaled_dot_product_attention(
                        _qs,
                        _ks,
                        _vs,
                        is_causal=_is_causal,
                        dropout_p=self.dropout.p if self.training else 0.0,
                    )
                    _sdpa_out = (
                        _sdpa_out.to(hidden_states.dtype)
                        .transpose(1, 2)
                        .contiguous()
                        .view(B, L, self.all_head_size)
                    )
                    _sdpa_out = torch.nan_to_num(_sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4)
                    if self.is_nested_encoder:
                        return _sdpa_out * self.output_scale, None
                    out = self.out_proj(_sdpa_out)
                    return out * self.output_scale, None
                except Exception:
                    pass

            # Dense manual
            scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale
            if is_causal:
                causal_mask = (
                    torch.triu(
                        torch.full((L, L_kv), float("-inf"), device=device, dtype=torch.float32),
                        diagonal=1,
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                scores = scores + causal_mask

            if valid_mask is not None:
                scores = scores.masked_fill(~valid_mask[:, None, None, :], -1e4)

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            attn_probs = F.softmax(scores, dim=-1).to(dtype)
            attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
            attn_probs = self.dropout(attn_probs)

            # MB-FIX-09: head_mask shape
            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.view(head_mask.shape[0], head_mask.shape[1], 1, 1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(1)  # interpret as (B, H, 1)
                attn_probs = attn_probs * head_mask.to(attn_probs.dtype)

            context = torch.matmul(attn_probs, V)
            out = context.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
        else:
            # ---- Block-sparse path ----
            pad = (self.block_size - L % self.block_size) % self.block_size
            if pad > 0:
                Q_pad = F.pad(Q, (0, 0, 0, pad))
                K_pad = F.pad(K, (0, 0, 0, pad))
                V_pad = F.pad(V, (0, 0, 0, pad))
                if valid_mask is not None:
                    valid_mask_padded = F.pad(valid_mask, (0, pad), value=False)
                else:
                    valid_mask_padded = torch.ones(B, L + pad, device=device, dtype=torch.bool)
            else:
                Q_pad, K_pad, V_pad = Q, K, V
                if valid_mask is not None:
                    valid_mask_padded = valid_mask
                else:
                    valid_mask_padded = torch.ones(B, L, device=device, dtype=torch.bool)

            Lp = L + pad
            nb = Lp // self.block_size
            bs = self.block_size

            Qb = Q_pad.view(B, H, nb, bs, d)
            Kb = K_pad.view(B, H, nb, bs, d)
            Vb = V_pad.view(B, H, nb, bs, d)

            # ---- Block-level scores ----
            Q_mean = Qb.mean(dim=3)
            K_mean = Kb.mean(dim=3)
            block_scores = (
                torch.matmul(Q_mean.float(), K_mean.float().transpose(-2, -1).contiguous()) * scale
            )

            # MB-FIX-08: block-0 preservation
            # - For bidirectional encoders, block 0 (CLS) should always be a candidate.
            # - For causal decoders, causality is enforced per-position below, so we
            #   don't need to force block 0; a query at block i > 0 would never see
            #   block 0 through causal masking.
            if not is_causal:
                block_scores_for_topk = block_scores.clone()
                block_scores_for_topk[:, :, :, 0] = block_scores_for_topk[:, :, :, 0] + 1e4
            else:
                block_scores_for_topk = block_scores.clone()
                # For causal: a query in block i can only attend to blocks j ≤ i.
                # Mask blocks j > i to -inf.
                causal_block = torch.triu(
                    torch.full((nb, nb), float("-inf"), device=device, dtype=torch.float32),
                    diagonal=1,
                )
                block_scores_for_topk = block_scores_for_topk + causal_block.unsqueeze(0).unsqueeze(
                    0
                )

            top_k = min(self.top_k, nb)
            _, top_idx = block_scores_for_topk.topk(top_k, dim=-1)  # (B, H, nb, top_k)

            # MB-FIX-05: valid block mask reshaped correctly
            valid_b_exp = valid_mask_padded.view(B, 1, nb, bs).expand(-1, H, -1, -1)

            out_blocks = torch.zeros_like(Qb)
            for i in range(nb):
                idx = top_idx[:, :, i, :]  # (B, H, top_k)
                idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, bs, d)
                K_sel = torch.gather(Kb, 2, idx_exp).view(B, H, top_k * bs, d)
                V_sel = torch.gather(Vb, 2, idx_exp).view(B, H, top_k * bs, d)

                idx_mask_exp = idx.unsqueeze(-1).expand(-1, -1, -1, bs)
                k_valid = torch.gather(valid_b_exp, 2, idx_mask_exp).reshape(B, H, 1, top_k * bs)

                Q_i = Qb[:, :, i]  # (B, H, bs, d)
                scores_i = (
                    torch.matmul(Q_i.float(), K_sel.float().transpose(-2, -1).contiguous()) * scale
                )
                scores_i = scores_i.masked_fill(~k_valid, -1e4)

                if is_causal:
                    q_pos = (i * bs + torch.arange(bs, device=device)).view(1, 1, bs, 1)
                    # absolute key positions for each of the top_k blocks
                    k_offsets = idx.unsqueeze(-1) * bs + torch.arange(bs, device=device).view(
                        1, 1, 1, bs
                    )
                    k_pos = k_offsets.reshape(B, H, 1, top_k * bs)
                    invalid = k_pos > q_pos
                    scores_i = scores_i.masked_fill(invalid, -1e4)

                scores_i = torch.nan_to_num(scores_i, nan=-1e4, posinf=1e4, neginf=-1e4)
                attn_i = F.softmax(scores_i, dim=-1).to(dtype)
                attn_i = torch.nan_to_num(attn_i, nan=0.0)
                attn_i = self.dropout(attn_i)
                out_blocks[:, :, i] = torch.matmul(attn_i, V_sel)

            out_pad = out_blocks.view(B, H, Lp, d)[:, :, :L, :]
            out = out_pad.permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)

        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            final_out = out
        else:
            final_out = self.out_proj(out)

        # MB-FIX-01: apply output_scale
        final_out = final_out * self.output_scale

        return final_out, None

    # ------------------------------------------------------------------
    def storage_bytes(self) -> dict[str, int]:
        def _sz(p):
            if p is None:
                return 0
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0

        out = {}
        for name, m in (
            ("q_proj", self.q_proj),
            ("k_proj", self.k_proj),
            ("v_proj", self.v_proj),
            ("out_proj", self.out_proj),
        ):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["output_scale"] = _sz(self.output_scale)
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("moba_inject")
@ATTENTION_REGISTRY.register("inject_moba")
def inject_moba(
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
        logger.warning("inject_moba: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # ---- Whole-model traversal ----
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
            res = inject_moba(
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

    # ---- Single-layer ----
    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if isinstance(cand, nn.Module) and cand is not target:
                        src = cand
                        break
        if src is None:
            src = target

        # Double-injection guard
        if isinstance(src, MoBAAttention) or "MoBAAttention" in src.__class__.__name__:
            return None
        if "DeltaNet" in src.__class__.__name__ or "LinearAttention" in src.__class__.__name__:
            return None

        # MB-FIX-10: single config assignment
        if model_config is None and hasattr(src, "config"):
            model_config = src.config
        if model_config is None and hasattr(target, "config"):
            model_config = target.config
        if model_config is None:

            class _Cfg:
                pass

            model_config = _Cfg()
            model_config.hidden_size = 768
            model_config.num_attention_heads = 12
            model_config.head_dim = 64
            model_config.model_type = ""

        block_size = int(
            kwargs.pop(
                "block_size",
                getattr(comp_config, "block_size", 64) if comp_config else 64,
            )
        )
        top_k = int(kwargs.pop("top_k", getattr(comp_config, "top_k", 4) if comp_config else 4))
        lora_rank = int(
            kwargs.pop(
                "lora_rank",
                getattr(comp_config, "lora_rank", 16) if comp_config else 16,
            )
        )
        lora_alpha = float(
            kwargs.pop(
                "lora_alpha",
                getattr(comp_config, "lora_alpha", 32.0) if comp_config else 32.0,
            )
        )

        new_attn = MoBAAttention(
            config=model_config,
            block_size=block_size,
            top_k=top_k,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            pretrained_attn=src if src is not target else None,
            **kwargs,
        )

        # Splice
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                target.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                target.attention = new_attn
        elif hasattr(target, "self_attn"):
            new_attn.is_nested_encoder = False
            target.self_attn = new_attn
        elif hasattr(target, "attn"):
            new_attn.is_nested_encoder = False
            target.attn = new_attn
        elif hasattr(target, "self"):
            new_attn.is_nested_encoder = True
            target.self = new_attn
        else:
            new_attn.is_nested_encoder = False
            target.attention = new_attn

        return new_attn
    except Exception as e:
        logger.exception("inject_moba failed at layer %d: %s", layer_idx, e)
        return None


# Aliases
inject_moba_attention = inject_moba

__all__ = [
    "MoBAAttention",
    "LoRALinear",
    "apply_rotary_pos_emb",
    "inject_moba",
    "inject_moba_attention",
]
