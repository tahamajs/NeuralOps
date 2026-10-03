"""
NASC: NEXUS-AEGIS Spectral Cascade Attention.
Hardened for multi-architecture production deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.

NOTE: This module is LoRA-only. The rank argument is accepted but not used for
SVD compression. Use lora_rank to control adapter size.
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
        q_rot_embed = (q_rot * cos.to(q_rot.dtype)) + (
            _rotate_half(q_rot) * sin.to(q_rot.dtype)
        )
        k_rot_embed = (k_rot * cos.to(k_rot.dtype)) + (
            _rotate_half(k_rot) * sin.to(k_rot.dtype)
        )
        q = (
            torch.cat([q_rot_embed, q_pass], dim=-1)
            if q_pass.numel() > 0
            else q_rot_embed
        )
        k = (
            torch.cat([k_rot_embed, k_pass], dim=-1)
            if k_pass.numel() > 0
            else k_rot_embed
        )
    else:
        cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
        q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
        k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))

    return q, k


def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    """Stable softpick function with negative-clamp and softmax fallback."""
    orig_dtype = x.dtype
    x = x.float()
    m = x.max(dim=dim, keepdim=True).values
    m = torch.where(m < -1e3, torch.zeros_like(m), m)
    z = x - m
    exp_z = torch.exp(z / temperature)
    exp_neg_m = torch.exp((-m) / temperature).clamp(max=1e4)
    p = F.relu(exp_z - exp_neg_m)
    denom = p.sum(dim=dim, keepdim=True)
    zero_mask = denom <= 1e-8
    if zero_mask.any():
        fallback = F.softmax(x, dim=dim)
        p = torch.where(zero_mask, fallback, p / (denom + 1e-8))
    else:
        p = p / denom
    return p.to(orig_dtype)


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty((lora_rank, in_features), **factory_kwargs)
            )
            self.lora_B = nn.Parameter(
                torch.empty((out_features, lora_rank), **factory_kwargs)
            )
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self) -> None:
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

    def freeze_base_weights(self) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            out = (
                out
                + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            )
        return out


@ATTENTION_REGISTRY.register("nasc", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("nasc_attention", supports_causal=True, supports_gqa=True)
class NASCAttention(BaseAttention):
    r"""NASC: NEXUS-AEGIS Spectral Cascade Attention.
    Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    """

    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any,
        rank: int | None = 16,
        window_size: int = 64,
        block_size: int = 32,
        top_k: int = 2,
        rff_features: int = 64,
        pretrained_attn: nn.Module | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.hidden_size: int = getattr(
            config, "hidden_size", kwargs.get("hidden_size", 768)
        )
        self.num_heads: int = getattr(
            config, "num_attention_heads", kwargs.get("num_attention_heads", 4)
        )

        cfg_head_dim = getattr(config, "head_dim", None) if config else None
        if cfg_head_dim is not None:
            self.head_dim = int(cfg_head_dim)
        elif kwargs.get("head_dim") is not None:
            self.head_dim = int(kwargs["head_dim"])
        elif self.num_heads > 0:
            self.head_dim = self.hidden_size // self.num_heads
        else:
            self.head_dim = 64
        self.all_head_size: int = self.num_heads * self.head_dim

        cfg_kv = (
            getattr(
                config, "num_key_value_heads", getattr(config, "num_kv_heads", None)
            )
            if config
            else None
        )
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat: int = self.num_heads // self.num_kv_heads
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        self.window_size: int = window_size
        self.block_size: int = block_size
        self.top_k: int = top_k
        self.rff_features: int = rff_features
        self.lora_rank: int = lora_rank
        self.lora_alpha: float = lora_alpha
        self.enable_cascade_on_encoders: bool = bool(
            kwargs.get(
                "enable_cascade_on_encoders",
                getattr(config, "enable_cascade_on_encoders", True),
            )
        )

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            m_type = str(getattr(config, "model_type", "")).lower() if config else ""
            self.is_decoder = (
                getattr(config, "is_decoder", False)
                or getattr(config, "causal", False)
                or any(
                    m_type.startswith(p)
                    for p in (
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
                )
                if config
                else False
            )

        self.is_nested_encoder: bool = False

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

        rff_init = torch.empty(self.head_dim, rff_features)
        nn.init.orthogonal_(rff_init)
        self.rff_proj = nn.Parameter(rff_init, requires_grad=False)
        self.gate_proj = nn.Linear(self.hidden_size, 3)

        dropout_p = (
            getattr(
                config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)
            )
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    @property
    def query(self) -> LoRALinear:
        return self.q_proj

    @property
    def key(self) -> LoRALinear:
        return self.k_proj

    @property
    def value(self) -> LoRALinear:
        return self.v_proj

    def copy_weights(self, src: Any) -> None:
        from neuralops.core.adapters import UniversalAdapter

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. UniversalAdapter extraction
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug("UniversalAdapter extraction fallback for %s: %s", proj, e)

        # 2. GPT-2 Fused c_attn Conv1D handling
        if hasattr(src, "c_attn") and (
            extracted["q"][0] is None
            or extracted["k"][0] is None
            or extracted["v"][0] is None
        ):
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            weight = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                bias = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                bias_data = bias.data.clone() if bias is not None else None
                c_cls = getattr(base_c, "__class__", type(base_c)).__name__
                if (
                    ("Conv1D" in c_cls)
                    or hasattr(base_c, "nf")
                    or (
                        "Conv1D"
                        in getattr(c_attn_mod, "__class__", type(c_attn_mod)).__name__
                    )
                    or hasattr(c_attn_mod, "nf")
                    or (weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size)
                ):
                    weight = weight.t().contiguous()
                split_size = weight.shape[0] // 3
                q_w = weight[:split_size, :]
                k_w = weight[split_size : 2 * split_size, :]
                v_w = weight[2 * split_size :, :]
                q_b = bias_data[:split_size] if bias_data is not None else None
                k_b = (
                    bias_data[split_size : 2 * split_size]
                    if bias_data is not None
                    else None
                )
                v_b = bias_data[2 * split_size :] if bias_data is not None else None
                extracted["q"] = (q_w, q_b)
                extracted["k"] = (k_w, k_b)
                extracted["v"] = (v_w, v_b)

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            c_proj_mod = src.c_proj
            base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
            w_o = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                b_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                p_cls = getattr(base_o, "__class__", type(base_o)).__name__
                if (
                    ("Conv1D" in p_cls)
                    or hasattr(base_o, "nf")
                    or (
                        "Conv1D"
                        in getattr(c_proj_mod, "__class__", type(c_proj_mod)).__name__
                    )
                    or hasattr(c_proj_mod, "nf")
                    or (w_o.ndim == 2 and w_o.shape[0] < w_o.shape[1])
                ):
                    w_o = w_o.t().contiguous()
                extracted["o"] = (w_o, b_o.data.clone() if b_o is not None else None)

        # 3. RoBERTa / BERT Nested Encoder handling
        if hasattr(src, "self") and (
            extracted["q"][0] is None
            or extracted["k"][0] is None
            or extracted["v"][0] is None
        ):
            s = src.self
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if (
            hasattr(src, "output")
            and hasattr(src.output, "dense")
            and extracted["o"][0] is None
        ):
            dense = src.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        # 4. Modern Decoders (Qwen3.5, LLaMA, Mistral) & DistilBERT
        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(src, name):
                        layer = getattr(src, name)
                        if hasattr(layer, "weight"):
                            extracted[proj] = (
                                layer.weight.data,
                                getattr(layer, "bias", None),
                            )
                            break

        # Detect nested encoder architecture
        cls_name = src.__class__.__name__
        has_out_proj = any(
            hasattr(src, attr)
            for attr in (
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
        if hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj"):
            self.is_nested_encoder = False
        elif (
            (
                hasattr(src, "self")
                and hasattr(src.self, "query")
                and hasattr(src, "output")
                and hasattr(src.output, "dense")
            )
            or (
                cls_name
                in (
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                )
                or (cls_name.endswith("SelfAttention") and not has_out_proj)
            )
            and not has_out_proj
        ):
            self.is_nested_encoder = True
        else:
            self.is_nested_encoder = False

        # Check required weights
        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning(
                "Module copy_weights warning. Proceeding with initialized weights."
            )

        # Frobenius norm validation
        for proj_key in ["q", "k", "v"]:
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    logger.warning(
                        f"NASCAttention extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f})"
                    )

        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                logger.warning(
                    f"NASCAttention extracted weight for output projection has near-zero Frobenius norm ({norm_o.item():.6f})"
                )

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        # Copy into projection modules
        with torch.no_grad():
            for proj_key, target in [
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
                ("o", self.out_proj),
            ]:
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
                            target_w.data.copy_(
                                w_oriented.view(out_d, rep, -1).mean(dim=1)
                            )
                            if b is not None and target.bias is not None:
                                target.bias.data.copy_(b.view(out_d, rep).mean(dim=1))
                        elif src_out < out_d and out_d % src_out == 0:
                            rep = out_d // src_out
                            target_w.data.copy_(w_oriented.repeat(rep, 1))
                            if b is not None and target.bias is not None:
                                target.bias.data.copy_(b.repeat(rep))
                        else:
                            target_w.data[
                                : min(out_d, src_out), : min(in_d, src_in)
                            ].copy_(
                                w_oriented[: min(out_d, src_out), : min(in_d, src_in)]
                            )
                            if b is not None and target.bias is not None:
                                target.bias.data[: min(len(target.bias), len(b))].copy_(
                                    b[: min(len(target.bias), len(b))]
                                )
                    else:
                        target_w.data[: min(out_d, src_out), : min(in_d, src_in)].copy_(
                            w_oriented[: min(out_d, src_out), : min(in_d, src_in)]
                        )
                        if b is not None and target.bias is not None:
                            b_val = b.data if isinstance(b, nn.Parameter) else b
                            target.bias.data[: min(len(target.bias), len(b_val))].copy_(
                                b_val[: min(len(target.bias), len(b_val))]
                            )
                elif proj_key == "o" and self.is_nested_encoder:
                    min_d = min(target.weight.shape[0], target.weight.shape[1])
                    target.weight.data.zero_()
                    target.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                    if target.bias is not None:
                        target.bias.data.zero_()

                target.freeze_base_weights()

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

    def _local_window_attention(self, Q, K, V, valid_mask, is_causal):
        B, H, L, d = Q.shape
        scale = 1.0 / math.sqrt(d)
        scores = (
            torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale
        )

        idx = torch.arange(L, device=Q.device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        mask = dist > self.window_size
        if is_causal:
            # Query i attending to key j: future keys (j > i) must be masked
            mask = mask | (idx.unsqueeze(0) > idx.unsqueeze(1))
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -10000.0)

        # Padding mask
        pad_k = (~valid_mask).unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(pad_k, -10000.0)
        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)

        attn = F.softmax(scores, dim=-1).to(Q.dtype)
        attn = self.dropout(attn)
        return torch.matmul(attn, V)

    def _sparse_block_attention(self, Q, K, V, valid_mask, is_causal):
        B, H, L, d = Q.shape
        pad_len = (self.block_size - L % self.block_size) % self.block_size
        if pad_len > 0:
            Q_pad = F.pad(Q, (0, 0, 0, pad_len))
            K_pad = F.pad(K, (0, 0, 0, pad_len))
            V_pad = F.pad(V, (0, 0, 0, pad_len))
            valid_pad = F.pad(valid_mask, (0, pad_len), value=False)
        else:
            Q_pad, K_pad, V_pad = Q, K, V
            valid_pad = valid_mask
        Lp = L + pad_len
        nb = Lp // self.block_size

        Qb = Q_pad.view(B, H, nb, self.block_size, d)
        Kb = K_pad.view(B, H, nb, self.block_size, d)
        Vb = V_pad.view(B, H, nb, self.block_size, d)
        valid_b = valid_pad.view(B, nb, self.block_size)

        Q_mean = Qb.mean(dim=3)
        K_mean = Kb.mean(dim=3)
        scale = 1.0 / math.sqrt(d)
        blk_scores = (
            torch.matmul(Q_mean.float(), K_mean.float().transpose(-2, -1).contiguous())
            * scale
        )

        if is_causal:
            causal_blk_mask = torch.triu(
                torch.full(
                    (nb, nb), float("-inf"), device=Q.device, dtype=torch.float32
                ),
                diagonal=1,
            )
            blk_scores = blk_scores + causal_blk_mask.unsqueeze(0).unsqueeze(0)

        top_k = min(self.top_k, nb)
        _, top_idx = blk_scores.topk(top_k, dim=-1)
        if is_causal:
            for i in range(nb):
                top_idx[:, :, i, 0] = i
        else:
            top_idx[:, :, :, 0] = 0

        out_blocks = torch.zeros_like(Qb)
        for i in range(nb):
            idx = top_idx[:, :, i, :]
            idx_exp = (
                idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.block_size, d)
            )
            K_sel = torch.gather(Kb, 2, idx_exp).reshape(
                B, H, top_k * self.block_size, d
            )
            V_sel = torch.gather(Vb, 2, idx_exp).reshape(
                B, H, top_k * self.block_size, d
            )

            scores_i = (
                torch.matmul(
                    Qb[:, :, i].float(), K_sel.float().transpose(-2, -1).contiguous()
                )
                * scale
            )

            if is_causal:
                q_pos = (
                    i * self.block_size + torch.arange(self.block_size, device=Q.device)
                ).view(1, 1, self.block_size, 1)
                k_pos = (
                    idx.unsqueeze(-1) * self.block_size
                    + torch.arange(self.block_size, device=Q.device).view(
                        1, 1, 1, self.block_size
                    )
                ).view(B, H, 1, top_k * self.block_size)
                scores_i = scores_i.masked_fill(k_pos > q_pos, -10000.0)

            valid_b_exp = valid_b.unsqueeze(1).expand(B, H, nb, self.block_size)
            idx_v = idx.unsqueeze(-1).expand(-1, -1, -1, self.block_size)
            valid_sel = torch.gather(valid_b_exp, 2, idx_v).reshape(
                B, H, top_k * self.block_size
            )
            pad_sel = ~valid_sel.unsqueeze(2)
            scores_i = scores_i.masked_fill(pad_sel, -10000.0)
            scores_i = torch.nan_to_num(scores_i, nan=-1e4, posinf=1e4, neginf=-1e4)

            attn_i = softpick(scores_i, dim=-1).to(Q.dtype)
            out_blocks[:, :, i] = torch.matmul(attn_i, V_sel)

        out = out_blocks.view(B, H, Lp, d)[:, :, :L, :]
        return out

    def _global_spectral_rff_attention(self, Q, K, V, valid_mask, is_causal):
        B, H, L, d = Q.shape
        pad = (~valid_mask).unsqueeze(1).unsqueeze(-1)

        Qf = torch.matmul(Q.float(), self.rff_proj.float())
        Kf = torch.matmul(K.float(), self.rff_proj.float())
        norm = 1.0 / math.sqrt(self.rff_features)
        phi_Q = torch.cat([torch.cos(Qf), torch.sin(Qf)], dim=-1) * norm
        phi_K = torch.cat([torch.cos(Kf), torch.sin(Kf)], dim=-1) * norm

        phi_K = phi_K.masked_fill(pad, 0.0)
        V_masked = V.float().masked_fill(pad, 0.0)

        if is_causal:
            # Causal linear recurrence via cumulative prefix sums
            KV = phi_K.unsqueeze(-1) * V_masked.unsqueeze(-2)  # [B, H, L, 2F, d]
            KV_cumsum = torch.cumsum(KV, dim=2)
            num = torch.matmul(phi_Q.unsqueeze(-2), KV_cumsum).squeeze(
                -2
            )  # [B, H, L, d]
            Z = torch.cumsum(phi_K, dim=2)
            den = (phi_Q * Z).sum(dim=-1, keepdim=True) + 1e-5
            return (num / den).to(Q.dtype)
        else:
            KV = torch.matmul(phi_K.transpose(-2, -1).contiguous(), V_masked)
            Z = phi_K.sum(dim=-2, keepdim=True)
            num = torch.matmul(phi_Q, KV)
            den = torch.matmul(phi_Q, Z.transpose(-2, -1).contiguous()) + 1e-5
            return (num / den).to(Q.dtype)

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

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        L_kv = kv_input.shape[1]

        is_causal = bool(self.is_decoder and not is_cross and L > 1)

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is None:
            valid_mask = torch.ones(
                B, L_kv, device=hidden_states.device, dtype=torch.bool
            )
        elif eff_mask.dim() == 2:
            valid_mask = (
                eff_mask.bool() if eff_mask.dtype == torch.bool else eff_mask > 0
            )
        elif eff_mask.dim() == 4:
            valid_mask = (
                eff_mask[:, 0, 0, :] > -10000.0
                if eff_mask.min() < -1000
                else eff_mask[:, 0, 0, :] > 0
            )
        else:
            valid_mask = torch.ones(
                B, L_kv, device=hidden_states.device, dtype=torch.bool
            )

        Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        K = (
            self.k_proj(kv_input)
            .view(B, L_kv, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )
        V = (
            self.v_proj(kv_input)
            .view(B, L_kv, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if self.num_kv_heads < self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        use_dense = bool(
            is_cross or (not self.is_decoder and not self.enable_cascade_on_encoders)
        )

        if not hasattr(self, "_logged_once"):
            logger.debug(
                "NASCAttention: L=%d, L_kv=%d, is_causal=%s, head_dim=%d, all_head_size=%d, path=%s",
                L,
                L_kv,
                is_causal,
                self.head_dim,
                self.all_head_size,
                "dense" if use_dense else "cascade",
            )
            self._logged_once = True

        if use_dense:
            # Fall back to standard scaled dot product attention for short/cross sequences or encoders
            scale = 1.0 / math.sqrt(d)
            scores = (
                torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous())
                * scale
            )
            if is_causal:
                causal_mask = (
                    torch.triu(
                        torch.full(
                            (L, L_kv),
                            float("-inf"),
                            device=hidden_states.device,
                            dtype=torch.float32,
                        ),
                        diagonal=1,
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                scores = scores + causal_mask

            if eff_mask is not None:
                if eff_mask.dim() == 2:
                    scores = (
                        scores
                        + (eff_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
                    )
                elif eff_mask.dim() == 3:
                    scores = scores + eff_mask.unsqueeze(1).float()
                elif eff_mask.dim() == 4:
                    scores = scores + eff_mask.float()

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            attn = F.softmax(scores, dim=-1).to(Q.dtype)
            attn = self.dropout(attn)
            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(-1)
                attn = attn * head_mask
            O = torch.matmul(attn, V)
        else:
            summary = hidden_states.mean(dim=1)
            gates = F.softmax(self.gate_proj(summary), dim=-1)  # [B, 3]
            g0 = gates[:, 0].view(B, 1, 1, 1)
            g1 = gates[:, 1].view(B, 1, 1, 1)
            g2 = gates[:, 2].view(B, 1, 1, 1)

            O_loc = self._local_window_attention(Q, K, V, valid_mask, is_causal)
            O_sp = self._sparse_block_attention(Q, K, V, valid_mask, is_causal)
            O_glb = self._global_spectral_rff_attention(Q, K, V, valid_mask, is_causal)

            O = g0 * O_loc + g1 * O_sp + g2 * O_glb

            if head_mask is not None:
                if head_mask.dim() == 1:
                    hm = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    hm = head_mask.unsqueeze(1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    hm = head_mask.unsqueeze(-1)
                else:
                    hm = head_mask
                O = O * hm

        context = O.permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        return out, None


@ATTENTION_REGISTRY.register("nasc_injection")
@ATTENTION_REGISTRY.register("inject_nasc")
@ATTENTION_REGISTRY.register("inject_nasc_attention")
def inject_nasc(
    layer: nn.Module | None = None,
    model: nn.Module | None = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 1,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
) -> NASCAttention | nn.Module | None:
    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # 1. Whole model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
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
            res = inject_nasc(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # 2. Single layer injection
    try:
        src_attn = old_attn
        if src_attn is None and target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if hasattr(cand, "self"):
                        src_attn = cand.self
                    else:
                        src_attn = cand
                    break

        if src_attn is None and target is not None and hasattr(target, "forward"):
            src_attn = target

        if src_attn is None:
            return None

        # Double-injection guard
        if isinstance(src_attn, NASCAttention):
            return None
        if "NASC" in src_attn.__class__.__name__:
            return None

        # Skip Qwen3.5 hybrid DeltaNet/LinearAttention layers
        c_name = src_attn.__class__.__name__
        if "DeltaNet" in c_name or "LinearAttention" in c_name:
            return None

        if cfg is None and hasattr(src_attn, "config"):
            cfg = src_attn.config

        if cfg is None:
            cfg = type("Cfg", (), {"hidden_size": 64, "num_heads": 4, "rank": 32})()

        rank = (
            getattr(comp_config, "rank", 16)
            if comp_config is not None
            else kwargs.pop("rank", 16)
        )
        window_size = (
            getattr(comp_config, "window_size", 64)
            if comp_config is not None
            else kwargs.pop("window_size", 64)
        )
        block_size = (
            getattr(comp_config, "block_size", 32)
            if comp_config is not None
            else kwargs.pop("block_size", 32)
        )
        top_k = (
            getattr(comp_config, "top_k", 2)
            if comp_config is not None
            else kwargs.pop("top_k", 2)
        )
        rff_features = (
            getattr(comp_config, "rff_features", 64)
            if comp_config is not None
            else kwargs.pop("rff_features", 64)
        )

        kwargs.pop("rank", None)
        kwargs.pop("window_size", None)
        kwargs.pop("block_size", None)
        kwargs.pop("top_k", None)
        kwargs.pop("rff_features", None)

        new_attn = NASCAttention(
            config=cfg,
            rank=rank,
            window_size=window_size,
            block_size=block_size,
            top_k=top_k,
            rff_features=rff_features,
            pretrained_attn=src_attn,
            **kwargs,
        )

        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

        if target is not None:
            if hasattr(target, "attention") and hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                target.attention.self = new_attn
                return new_attn
            else:
                new_attn.is_nested_encoder = False
                if hasattr(target, "attention"):
                    target.attention = new_attn
                    return new_attn
                elif hasattr(target, "self"):
                    new_attn.is_nested_encoder = True
                    target.self = new_attn
                    return new_attn
                elif hasattr(target, "self_attn"):
                    target.self_attn = new_attn
                    return new_attn
                elif hasattr(target, "attn"):
                    target.attn = new_attn
                    return new_attn
            target.attention = new_attn
            return new_attn
        return new_attn
    except Exception as e:
        logger.error("inject_nasc failed: %s", e)
        return None


# Aliases
inject_nasc_attention = inject_nasc
