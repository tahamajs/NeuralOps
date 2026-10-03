"""
MixAttention-MoSKA: Frequency/Variance-Aware Sparse Attention (FASA).
Production-Hardened (v2).

"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import get_token_valid_mask
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


def apply_rotary_pos_emb(q, k, cos, sin):
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
# Main attention
# ===========================================================================
@ATTENTION_REGISTRY.register("fasa", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("fasa_attention", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("mixattention_moska", supports_causal=True, supports_gqa=True)
class MixAttentionMoSKA(BaseAttention):
    """
    MixAttention-MoSKA (FASA): frequency/variance-aware sparse attention.

    For non-causal self-attention and encoder-style models, we prune the K/V
    set down to the top `fasa_keep_ratio` tokens by variance importance,
    always preserving the [CLS] token at index 0.

    For causal decoders, cross-attention, and incremental decoding (when
    `past_key_value` is present), we bypass pruning entirely and run exact
    dense attention via SDPA / manual softmax.
    """

    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = 32,
        fasa_keep_ratio: float = 0.5,
        pretrained_attn: nn.Module | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        # FASA-FIX-04: config None-safe
        if config is None:

            class _Cfg:
                pass

            config = _Cfg()
            config.hidden_size = kwargs.get("hidden_size", 768)
            config.num_attention_heads = kwargs.get(
                "num_attention_heads", kwargs.get("num_heads", 12)
            )
            config.head_dim = kwargs.get("head_dim", 64)
            config.model_type = kwargs.get("model_type", "")
            config.is_decoder = kwargs.get("is_decoder", False)
            config.causal = kwargs.get("is_causal", False)

        try:
            super().__init__(config, rank if rank is not None else 32)
        except TypeError:
            super().__init__(config)

        self.dim = int(
            kwargs.get("hidden_size") or getattr(config, "hidden_size", getattr(config, "dim", 768))
        )
        self.hidden_size = self.dim

        n_heads = (
            kwargs.get("num_heads")
            or kwargs.get("num_attention_heads")
            or getattr(config, "num_attention_heads", getattr(config, "n_heads", 0))
        )
        # FASA-FIX-11
        if not n_heads or int(n_heads) <= 0:
            n_heads = max(1, self.dim // 64)
        self.num_heads = int(n_heads)

        cfg_head_dim = getattr(config, "head_dim", None)
        self.head_dim = int(
            cfg_head_dim or kwargs.get("head_dim") or (self.dim // max(1, self.num_heads))
        )
        self.all_head_size = self.num_heads * self.head_dim

        self.fasa_keep_ratio = float(fasa_keep_ratio)
        self.protect_first_token = bool(kwargs.get("protect_first_token", True))
        self.importance_method = str(kwargs.get("importance_method", "var"))

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

        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)

        # ---- decoder detection ----
        _mt = str(getattr(config, "model_type", "") or "").lower()
        _dec_prefixes = (
            "gpt",
            "qwen",
            "llama",
            "mistral",
            "gemma",
            "phi",
            "opt",
            "bloom",
            "falcon",
            "olmo",
            "mpt",
            "stablelm",
        )
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            self.is_decoder = bool(
                getattr(config, "is_decoder", False)
                or getattr(config, "causal", False)
                or any(_mt.startswith(p) for p in _dec_prefixes)
                or kwargs.get("is_causal", False)
            )
            # FASA-FIX-01: also detect GPT-2 by source attention
            if pretrained_attn is not None:
                if hasattr(pretrained_attn, "c_attn"):
                    self.is_decoder = True
                if hasattr(pretrained_attn, "is_decoder") and pretrained_attn.is_decoder:
                    self.is_decoder = True

        self.is_nested_encoder = False

        self.W_q = LoRALinear(
            self.dim,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.W_k = LoRALinear(
            self.dim,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.W_v = LoRALinear(
            self.dim,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        dropout_p = float(
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
        )
        self.dropout = nn.Dropout(dropout_p)

        # FASA-FIX-09: storage accounting
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ---- aliases ----
    @property
    def query(self):
        return self.W_q

    @property
    def key(self):
        return self.W_k

    @property
    def value(self):
        return self.W_v

    @property
    def q_proj(self):
        return self.W_q

    @property
    def k_proj(self):
        return self.W_k

    @property
    def v_proj(self):
        return self.W_v

    @property
    def q_lin(self):
        return self.W_q

    @property
    def k_lin(self):
        return self.W_k

    @property
    def v_lin(self):
        return self.W_v

    @property
    def out_lin(self):
        return self.out_proj

    @property
    def dense(self):
        return self.out_proj

    @property
    def o_proj(self):
        return self.out_proj

    @property
    def c_proj(self):
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
            except Exception:
                pass

        # 2. GPT-2 fused c_attn
        if hasattr(src, "c_attn") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            c_mod = src.c_attn
            base_c = getattr(c_mod, "base_layer", c_mod)
            weight = getattr(base_c, "weight", getattr(c_mod, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                bias = getattr(base_c, "bias", getattr(c_mod, "bias", None))
                bias_data = bias.data.clone() if bias is not None else None
                is_conv1d = (
                    "Conv1D" in c_mod.__class__.__name__
                    or "Conv1D" in base_c.__class__.__name__
                    or hasattr(c_mod, "nf")
                    or hasattr(base_c, "nf")
                    or (weight.dim() == 2 and weight.shape[0] < weight.shape[1])
                )
                if is_conv1d:
                    weight = weight.t().contiguous()
                split = weight.shape[0] // 3
                extracted["q"] = (
                    weight[:split],
                    bias_data[:split] if bias_data is not None else None,
                )
                extracted["k"] = (
                    weight[split : 2 * split],
                    bias_data[split : 2 * split] if bias_data is not None else None,
                )
                extracted["v"] = (
                    weight[2 * split :],
                    bias_data[2 * split :] if bias_data is not None else None,
                )

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            p_mod = src.c_proj
            base_o = getattr(p_mod, "base_layer", p_mod)
            w_o = getattr(base_o, "weight", getattr(p_mod, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                is_conv1d = (
                    "Conv1D" in p_mod.__class__.__name__
                    or "Conv1D" in base_o.__class__.__name__
                    or hasattr(p_mod, "nf")
                    or hasattr(base_o, "nf")
                    or (w_o.dim() == 2 and w_o.shape[0] < w_o.shape[1])
                )
                if is_conv1d:
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_o, "bias", getattr(p_mod, "bias", None))
                extracted["o"] = (w_o, b_o.data.clone() if b_o is not None else None)

        # 3. BERT / RoBERTa nested (FASA-FIX-02: extract .data, not module)
        if hasattr(src, "self") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            s = src.self
            for key, sub_name in (("q", "query"), ("k", "key"), ("v", "value")):
                if extracted[key][0] is None:
                    m = getattr(s, sub_name, None)
                    if m is not None and hasattr(m, "weight"):
                        bias = getattr(m, "bias", None)
                        extracted[key] = (
                            m.weight.data,
                            bias.data if bias is not None else None,
                        )

        if hasattr(src, "output") and hasattr(src.output, "dense") and extracted["o"][0] is None:
            d = getattr(src.output.dense, "base_layer", src.output.dense)
            if hasattr(d, "weight"):
                bias = getattr(d, "bias", None)
                extracted["o"] = (
                    d.weight.data,
                    bias.data if bias is not None else None,
                )

        # 4. Flat modules
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
                        if hasattr(m, "weight"):
                            b = getattr(m, "bias", None)
                            extracted[proj] = (
                                m.weight.data,
                                b.data if b is not None else None,
                            )
                            break

        # ---- nested encoder detection ----
        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out = any(hasattr(src, a) for a in ("dense", "out_proj", "out_lin", "c_proj", "o_proj"))
        is_nested = False
        if (
            cls_name
            in (
                "BertSelfAttention",
                "RobertaSelfAttention",
                "XLMRobertaSelfAttention",
                "ElectraSelfAttention",
                "DebertaSelfAttention",
            )
            or hasattr(src, "self")
            and hasattr(src, "output")
            or hasattr(src, "query")
            and not has_out
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning("MixAttentionMoSKA copy_weights: missing %s", missing)

        # Frobenius norm checks
        for k_ in ("q", "k", "v"):
            w = extracted[k_][0]
            if w is not None:
                n = torch.linalg.norm(w.float()).item()
                if n < 1e-4:
                    logger.warning("FASA: %s weight near-zero norm (%.2e)", k_, n)
        if extracted["o"][0] is not None:
            n = torch.linalg.norm(extracted["o"][0].float()).item()
            if n < 1e-4:
                logger.warning("FASA: o weight near-zero norm (%.2e)", n)

        def _orient(w, in_dim):
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        with torch.no_grad():
            for proj_key, target in (
                ("q", self.W_q),
                ("k", self.W_k),
                ("v", self.W_v),
                ("o", self.out_proj),
            ):
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.dim
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
                            m0 = min(out_d, src_out)
                            m1 = min(in_d, src_in)
                            target_w.data[:m0, :m1].copy_(w_oriented[:m0, :m1])
                            if b is not None and target.bias is not None:
                                mb = min(target.bias.shape[0], b.shape[0])
                                target.bias.data[:mb].copy_(b[:mb])
                    else:
                        m0 = min(out_d, src_out)
                        m1 = min(in_d, src_in)
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
            for m in (self.W_q, self.W_k, self.W_v, self.out_proj):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        # FASA-FIX-06: don't overwrite if src lacks is_decoder
        if hasattr(src, "is_decoder") and src.is_decoder:
            self.is_decoder = True

    # ------------------------------------------------------------------
    # FASA pruning
    # ------------------------------------------------------------------
    def _fasa_prune(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        mask_2d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        B, H, L, D_h = k.shape
        if L <= 8:
            return k, v, mask_2d, None

        method = getattr(self, "importance_method", "var")
        if method == "l2":
            importance = torch.norm(k.float(), p=2, dim=-1).mean(dim=1)  # (B, L)
        else:
            importance = k.float().var(dim=-1).mean(dim=1)  # (B, L)

        if mask_2d is not None:
            importance = importance.masked_fill(~mask_2d, -1e4)

        # FASA-FIX-03: clone before in-place modification
        importance = importance.clone()
        if self.protect_first_token:
            importance[:, 0] = 1e9

        keep = max(8, min(L, int(math.ceil(L * self.fasa_keep_ratio))))
        _, idx = torch.topk(importance, keep, dim=-1)
        idx, _ = torch.sort(idx, dim=-1)

        idx_exp = idx.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D_h)
        k_pruned = k.gather(2, idx_exp)
        v_pruned = v.gather(2, idx_exp)

        pruned_mask = None
        if mask_2d is not None:
            pruned_mask = mask_2d.gather(1, idx)

        return k_pruned, v_pruned, pruned_mask, idx

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_valid_mask(
        attention_mask: torch.Tensor | None,
        B: int,
        L: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Return a bool (B, L) mask where True = valid."""
        if attention_mask is None:
            return None

        def _to_bool(t):
            if t.dtype == torch.bool:
                return t
            if t.dtype.is_floating_point:
                return t > -1e4
            return t > 0

        if attention_mask.dim() == 2:
            m = _to_bool(attention_mask)
            if m.shape[-1] == L:
                return m
        elif attention_mask.dim() == 3:
            m = _to_bool(attention_mask).squeeze(1)
            if m.shape[-1] == L:
                return m
        elif attention_mask.dim() == 4:
            m = _to_bool(attention_mask).any(dim=-2).squeeze(1)
            if m.shape[-1] == L:
                return m

        # Fallback: try the framework helper
        try:
            m = get_token_valid_mask(attention_mask, B, L, device)
            if isinstance(m, torch.Tensor) and m.shape == (B, L):
                return m.bool()
        except Exception:
            pass
        return None

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
    ):
        # FASA-FIX-07: fold `mask` kwarg into attention_mask
        mask = kwargs.get("mask")
        if attention_mask is None and mask is not None:
            attention_mask = mask

        x = hidden_states
        B, L, D = x.shape
        H = self.num_heads
        device = x.device
        dtype = x.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else x
        L_kv = kv_input.shape[1]

        q = self.W_q(x).view(B, L, H, self.head_dim).permute(0, 2, 1, 3)
        k = self.W_k(kv_input).view(B, L_kv, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.W_v(kv_input).view(B, L_kv, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_heads < self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        scale = 1.0 / math.sqrt(self.head_dim)
        is_causal = bool(self.is_decoder and not is_cross and L > 1)

        # FASA-FIX-01: disable pruning when:
        #   - we are in a causal decoder,
        #   - we are doing cross-attention,
        #   - a KV-cache is present (incremental decoding),
        #   - hidden_states and kv_input have different lengths (mixed mode).
        use_pruning = not is_causal and not is_cross and past_key_value is None and L_kv == L

        if not use_pruning:
            # -------- Dense path (SDPA / manual) --------
            if eff_mask is None and head_mask is None and not output_attentions:
                try:
                    _qs = q.half() if q.dtype == torch.float32 else q
                    _ks = k.half() if k.dtype == torch.float32 else k
                    _vs = v.half() if v.dtype == torch.float32 else v
                    _sdpa_out = F.scaled_dot_product_attention(
                        _qs,
                        _ks,
                        _vs,
                        is_causal=is_causal,
                        dropout_p=self.dropout.p if self.training else 0.0,
                    )
                    _sdpa_out = (
                        _sdpa_out.to(dtype)
                        .transpose(1, 2)
                        .contiguous()
                        .view(B, L, self.all_head_size)
                    )
                    _sdpa_out = torch.nan_to_num(_sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4)
                    if self.is_nested_encoder:
                        return _sdpa_out, None
                    return self.out_proj(_sdpa_out), None
                except Exception:
                    pass

            scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale
            if is_causal:
                causal = (
                    torch.triu(
                        torch.full((L, L_kv), float("-inf"), device=device, dtype=torch.float32),
                        diagonal=1,
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                scores = scores + causal

            valid = self._normalize_valid_mask(eff_mask, B, L_kv, device)
            if valid is not None:
                scores = scores.masked_fill(~valid[:, None, None, :], -1e4)

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            attn_weights = F.softmax(scores, dim=-1).to(dtype)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            attn_weights = self.dropout(attn_weights)

            # FASA-FIX-10: unified head_mask handling
            if head_mask is not None:
                hm = head_mask.to(attn_weights.dtype)
                if hm.dim() == 1:
                    hm = hm.view(1, -1, 1, 1)
                elif hm.dim() == 2:
                    hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
                elif hm.dim() == 3:
                    hm = hm.unsqueeze(1)
                attn_weights = attn_weights * hm

            out = (
                torch.matmul(attn_weights, v)
                .transpose(1, 2)
                .contiguous()
                .view(B, L, self.all_head_size)
            )
            final_attn_weights = attn_weights if output_attentions else None
        else:
            # -------- FASA pruning path --------
            valid = self._normalize_valid_mask(eff_mask, B, L_kv, device)
            k_p, v_p, p_mask, kept_idx = self._fasa_prune(k, v, valid)

            attn_scores = (
                torch.matmul(q.float(), k_p.float().transpose(-2, -1).contiguous()) * scale
            )

            if p_mask is not None:
                attn_scores = attn_scores.masked_fill(~p_mask[:, None, None, :], -1e4)

            attn_scores = torch.nan_to_num(attn_scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            attn_weights = F.softmax(attn_scores, dim=-1).to(dtype)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            attn_weights = self.dropout(attn_weights)

            if head_mask is not None:
                hm = head_mask.to(attn_weights.dtype)
                if hm.dim() == 1:
                    hm = hm.view(1, -1, 1, 1)
                elif hm.dim() == 2:
                    hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
                elif hm.dim() == 3:
                    hm = hm.unsqueeze(1)
                attn_weights = attn_weights * hm

            out = (
                torch.matmul(attn_weights, v_p)
                .transpose(1, 2)
                .contiguous()
                .view(B, L, self.all_head_size)
            )

            if output_attentions and kept_idx is not None:
                full_weights = torch.zeros(B, H, L, L_kv, device=device, dtype=dtype)
                kept_exp = kept_idx.unsqueeze(1).unsqueeze(2).expand(-1, H, L, -1)
                final_attn_weights = full_weights.scatter(3, kept_exp, attn_weights)
            elif output_attentions:
                final_attn_weights = attn_weights
            else:
                final_attn_weights = None

        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = out
        else:
            output = self.out_proj(out)

        return output, final_attn_weights

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
            ("W_q", self.W_q),
            ("W_k", self.W_k),
            ("W_v", self.W_v),
            ("out_proj", self.out_proj),
        ):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("fasa_inject")
@ATTENTION_REGISTRY.register("mixattention_moska_inject")
@ATTENTION_REGISTRY.register("inject_mixattention_moska")
@ATTENTION_REGISTRY.register("inject_fasa")
def inject_mixattention_moska(
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
        logger.warning("inject_mixattention_moska: max recursion depth exceeded.")
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
            res = inject_mixattention_moska(
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

        # FASA-FIX-05: idempotency guard
        if (
            isinstance(src_attn, MixAttentionMoSKA)
            or "MixAttentionMoSKA" in src_attn.__class__.__name__
        ):
            return None
        if (
            "DeltaNet" in src_attn.__class__.__name__
            or "LinearAttention" in src_attn.__class__.__name__
        ):
            return None

        # ---- Build new attention ----
        keep_ratio = kwargs.pop(
            "fasa_keep_ratio",
            (getattr(comp_config, "fasa_keep_ratio", 0.5) if comp_config is not None else 0.5),
        )
        kwargs.pop("rank", None)

        new_attn = MixAttentionMoSKA(
            config=cfg or getattr(src_attn, "config", None),
            fasa_keep_ratio=keep_ratio,
            pretrained_attn=src_attn if src_attn is not target else None,
            **kwargs,
        )

        # Move to source device/dtype
        p = next(src_attn.parameters(), None) if src_attn is not target else None
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

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

        return new_attn
    except Exception as e:
        logger.exception("inject_mixattention_moska failed at layer %d: %s", layer_idx, e)
        return None


fasa_inject = inject_mixattention_moska
FASA = MixAttentionMoSKA

__all__ = [
    "MixAttentionMoSKA",
    "FASA",
    "fasa_inject",
    "inject_mixattention_moska",
    "apply_rotary_pos_emb",
    "LoRALinear",
]
