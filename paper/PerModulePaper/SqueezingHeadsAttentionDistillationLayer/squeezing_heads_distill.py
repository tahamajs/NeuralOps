"""
Squeezing Heads Attention Distillation Layer.
Multi-head attention wrapper designed for knowledge distillation and head capacity compression experiments.

Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    d = x.shape[-1]
    rotary_dim = 2 * (d // 2)
    x1 = x[..., : rotary_dim // 2]
    x2 = x[..., rotary_dim // 2 : rotary_dim]
    rotated = torch.cat((-x2, x1), dim=-1)
    if rotary_dim < d:
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) with partial dimension and independent seq-len support."""
    if cos is None or sin is None:
        return q, k

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    head_dim = q.shape[-1]
    rotary_dim = 2 * (min(cos.shape[-1], head_dim, k.shape[-1]) // 2)
    if rotary_dim == 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    cos_q = cos[..., : q.shape[-2], :rotary_dim].to(q.device, q.dtype)
    sin_q = sin[..., : q.shape[-2], :rotary_dim].to(q.device, q.dtype)

    cos_k = cos[..., : k.shape[-2], :rotary_dim].to(k.device, k.dtype)
    sin_k = sin[..., : k.shape[-2], :rotary_dim].to(k.device, k.dtype)

    q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
    k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)

    q_out = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot_embed
    k_out = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot_embed
    return q_out, k_out


class LoRALinear(nn.Module):
    """Linear projection with frozen base weights and trainable LoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(1, lora_rank)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register("squeezing_heads_distiller", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("squeezing_heads_distill", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("squeezing_heads", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("squeezing_heads_module", supports_causal=True, supports_gqa=True)
class SqueezingHeadsDistiller(BaseAttention):
    r"""Attention layer wrapper for multi-head distillation and compression studies.
    Hardened for production use across diverse transformer architectures.
    """

    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        old_attn: Optional[nn.Module] = None,
        rank: Optional[int] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)

        # 1. Dimensions & Architecture
        self.hidden_size: int = int(_cfg_get(
            config, "hidden_size", kwargs.get("hidden_size", 768)
        ))
        self.num_heads: int = int(_cfg_get(
            config,
            "num_attention_heads",
            _cfg_get(config, "num_heads", kwargs.get("num_attention_heads", 12)),
        ))
        cfg_hd = _cfg_get(config, "head_dim", None)
        kw_hd = kwargs.get("head_dim", None)
        if cfg_hd is not None:
            self.head_dim = int(cfg_hd)
        elif kw_hd is not None:
            self.head_dim = int(kw_hd)
        elif old_attn is not None:
            for attr in ("q_proj", "query", "q_lin"):
                if hasattr(old_attn, attr):
                    qw = getattr(old_attn, attr).weight
                    self.head_dim = qw.shape[0] // self.num_heads
                    break
            else:
                self.head_dim = self.hidden_size // self.num_heads if self.num_heads > 0 else 64
        else:
            self.head_dim = self.hidden_size // self.num_heads if self.num_heads > 0 else 64

        cfg_kv = _cfg_get(
            config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None)
        )
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat: int = max(1, self.num_heads // self.num_kv_heads)
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.q_head_size: int = self.num_heads * self.head_dim
        self.all_head_size: int = self.q_head_size

        # 2. Causal decoder flag detection
        _mt = str(_cfg_get(config, "model_type", "") or "").lower()
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        self.is_decoder = (
            bool(_cfg_get(config, "is_decoder", False))
            or bool(_cfg_get(config, "is_causal", False))
            or any(_mt.startswith(p) for p in _decoder_prefixes)
            or (old_attn is not None and hasattr(old_attn, "c_attn"))
            or bool(kwargs.get("is_causal", False))
        )
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        self.is_causal = self.is_decoder

        self.is_nested_encoder: bool = False
        self.lora_rank = int(lora_rank if lora_rank is not None else 16)
        self.lora_alpha = lora_alpha

        # 3. Projections with LoRA
        self.q_proj = LoRALinear(
            self.hidden_size,
            self.q_head_size,
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
            self.q_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        dropout_p = float(
            getattr(
                config,
                "dropout",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            ) if config else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        src_attn = old_attn if old_attn is not None else kwargs.get("pretrained_attn", None)
        if src_attn is not None:
            self.copy_weights(src_attn)

    @property
    def query(self) -> nn.Module:
        return self.q_proj

    @property
    def key(self) -> nn.Module:
        return self.k_proj

    @property
    def value(self) -> nn.Module:
        return self.v_proj

    @property
    def q_lin(self) -> nn.Module:
        return self.q_proj

    @property
    def k_lin(self) -> nn.Module:
        return self.k_proj

    @property
    def v_lin(self) -> nn.Module:
        return self.v_proj

    def copy_weights(self, old_attn: nn.Module) -> None:
        """Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA."""
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            # Detect nested encoder container or inner self-attention
            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            has_out_lin = hasattr(old_attn, "out_lin") or hasattr(old_attn, "c_proj") or hasattr(old_attn, "o_proj")
            if has_out_lin:
                self.is_nested_encoder = False
            elif (
                hasattr(old_attn, "self")
                and hasattr(getattr(old_attn, "self"), "query")
                and hasattr(old_attn, "output")
                and hasattr(getattr(old_attn, "output"), "dense")
            ):
                self.is_nested_encoder = True
            elif src_cls_name in ("BertSelfAttention", "RobertaSelfAttention"):
                self.is_nested_encoder = True
            else:
                self.is_nested_encoder = False

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    w_o = old_attn.output.dense.weight
                    b_o = old_attn.output.dense.bias
            else:
                src = old_attn

            # Pattern 1: GPT-2 Fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach()
                c_attn_b = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                p_cls = getattr(src.c_attn, "__class__", type(src.c_attn)).__name__
                if ("Conv1D" in p_cls) or hasattr(src.c_attn, "nf") or (c_attn_w.shape[0] == self.hidden_size and c_attn_w.shape[1] != self.hidden_size):
                    c_attn_w = c_attn_w.t().contiguous()

                H = self.hidden_size
                w_q = c_attn_w[:H, :]
                w_k = c_attn_w[H : 2 * H, :]
                w_v = c_attn_w[2 * H : 3 * H, :]

                if c_attn_b is not None:
                    b_q = c_attn_b[:H]
                    b_k = c_attn_b[H : 2 * H]
                    b_v = c_attn_b[2 * H : 3 * H]

                out_w = src.c_proj.weight.detach()
                p_proj_cls = getattr(src.c_proj, "__class__", type(src.c_proj)).__name__
                if ("Conv1D" in p_proj_cls) or hasattr(src.c_proj, "nf") or (out_w.shape[0] == self.hidden_size and out_w.shape[1] != self.hidden_size):
                    out_w = out_w.t().contiguous()
                w_o = out_w
                b_o = src.c_proj.bias.detach() if src.c_proj.bias is not None else None

            # Pattern 2: RoBERTa / BERT separate query, key, value
            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                w_q = src.query.weight.detach()
                b_q = src.query.bias.detach() if src.query.bias is not None else None
                w_k = src.key.weight.detach()
                b_k = src.key.bias.detach() if src.key.bias is not None else None
                w_v = src.value.weight.detach()
                b_v = src.value.bias.detach() if src.value.bias is not None else None

                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, attr):
                            proj = getattr(src, attr)
                            w_o = proj.weight.detach()
                            b_o = proj.bias.detach() if proj.bias is not None else None
                            break

            # Pattern 3: Qwen / LLaMA (q_proj, k_proj, v_proj, o_proj)
            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q = src.q_proj.weight.detach()
                b_q = src.q_proj.bias.detach() if src.q_proj.bias is not None else None
                w_k = src.k_proj.weight.detach()
                b_k = src.k_proj.bias.detach() if src.k_proj.bias is not None else None
                w_v = src.v_proj.weight.detach()
                b_v = src.v_proj.bias.detach() if src.v_proj.bias is not None else None
                if hasattr(src, "o_proj"):
                    w_o = src.o_proj.weight.detach()
                    b_o = src.o_proj.bias.detach() if src.o_proj.bias is not None else None

            # Pattern 4: DistilBERT (q_lin, k_lin, v_lin, out_lin)
            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                w_q = src.q_lin.weight.detach()
                b_q = src.q_lin.bias.detach() if src.q_lin.bias is not None else None
                w_k = src.k_lin.weight.detach()
                b_k = src.k_lin.bias.detach() if src.k_lin.bias is not None else None
                w_v = src.v_lin.weight.detach()
                b_v = src.v_lin.bias.detach() if src.v_lin.bias is not None else None
                if hasattr(src, "out_lin"):
                    w_o = src.out_lin.weight.detach()
                    b_o = src.out_lin.bias.detach() if src.out_lin.bias is not None else None

            # Fallback: UniversalAdapter
            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter(old_attn)
                    extracted = adapter.extract_qkv_out()
                    if extracted:
                        w_q = extracted.get("q_weight")
                        b_q = extracted.get("q_bias")
                        w_k = extracted.get("k_weight")
                        b_k = extracted.get("k_bias")
                        w_v = extracted.get("v_weight")
                        b_v = extracted.get("v_bias")
                        if w_o is None:
                            w_o = extracted.get("out_weight")
                            b_o = extracted.get("out_bias")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            # Verification of essential projections
            missing = []
            if w_q is None:
                missing.append("query/q_proj")
            if w_k is None:
                missing.append("key/k_proj")
            if w_v is None:
                missing.append("value/v_proj")
            if missing:
                logger.warning(f"Failed to extract required projections: {missing}")

            # Frobenius norm checks on source weights
            for name, w in [("w_q", w_q), ("w_k", w_k), ("w_v", w_v)]:
                if w is not None and torch.norm(w.float(), p="fro").item() < 1e-4:
                    logger.warning(f"Copied weight {name} has near-zero Frobenius norm (< 1e-4)")

            # Copy Q: preserve principal head dimensions
            self.q_proj.weight.zero_()
            m_q_out = min(self.q_head_size, w_q.shape[0])
            m_q_in = min(self.hidden_size, w_q.shape[1])
            self.q_proj.weight[:m_q_out, :m_q_in].copy_(w_q[:m_q_out, :m_q_in])
            if b_q is not None and self.q_proj.bias is not None:
                m_bq = min(self.q_proj.bias.shape[0], b_q.shape[0])
                self.q_proj.bias[:m_bq].copy_(b_q[:m_bq])

            # Copy K & V with GQA pooling or principal head slice
            if w_k.shape[0] == self.kv_head_size and w_k.shape[1] == self.hidden_size:
                self.k_proj.weight.copy_(w_k)
                if b_k is not None and self.k_proj.bias is not None:
                    self.k_proj.bias.copy_(b_k)
            elif w_k.shape[0] > self.kv_head_size and w_k.shape[0] % self.kv_head_size == 0 and w_k.shape[1] >= self.hidden_size:
                rep = w_k.shape[0] // self.kv_head_size
                w_k_pooled = (
                    w_k[:, :self.hidden_size].view(self.num_kv_heads, rep, self.head_dim, -1)
                    .mean(dim=1)
                    .view(self.kv_head_size, -1)
                )
                self.k_proj.weight.copy_(w_k_pooled)
                if b_k is not None and self.k_proj.bias is not None:
                    b_k_pooled = (
                        b_k.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
                    )
                    self.k_proj.bias.copy_(b_k_pooled)
            else:
                self.k_proj.weight.zero_()
                m_k_out = min(self.kv_head_size, w_k.shape[0])
                m_k_in = min(self.hidden_size, w_k.shape[1])
                self.k_proj.weight[:m_k_out, :m_k_in].copy_(w_k[:m_k_out, :m_k_in])
                if b_k is not None and self.k_proj.bias is not None:
                    m_bk = min(self.k_proj.bias.shape[0], b_k.shape[0])
                    self.k_proj.bias[:m_bk].copy_(b_k[:m_bk])

            if w_v.shape[0] == self.kv_head_size and w_v.shape[1] == self.hidden_size:
                self.v_proj.weight.copy_(w_v)
                if b_v is not None and self.v_proj.bias is not None:
                    self.v_proj.bias.copy_(b_v)
            elif w_v.shape[0] > self.kv_head_size and w_v.shape[0] % self.kv_head_size == 0 and w_v.shape[1] >= self.hidden_size:
                rep = w_v.shape[0] // self.kv_head_size
                w_v_pooled = (
                    w_v[:, :self.hidden_size].view(self.num_kv_heads, rep, self.head_dim, -1)
                    .mean(dim=1)
                    .view(self.kv_head_size, -1)
                )
                self.v_proj.weight.copy_(w_v_pooled)
                if b_v is not None and self.v_proj.bias is not None:
                    b_v_pooled = (
                        b_v.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
                    )
                    self.v_proj.bias.copy_(b_v_pooled)
            else:
                self.v_proj.weight.zero_()
                m_v_out = min(self.kv_head_size, w_v.shape[0])
                m_v_in = min(self.hidden_size, w_v.shape[1])
                self.v_proj.weight[:m_v_out, :m_v_in].copy_(w_v[:m_v_out, :m_v_in])
                if b_v is not None and self.v_proj.bias is not None:
                    m_bv = min(self.v_proj.bias.shape[0], b_v.shape[0])
                    self.v_proj.bias[:m_bv].copy_(b_v[:m_bv])

            # Copy Out projection if not nested encoder
            if self.is_nested_encoder:
                nn.init.eye_(self.out_proj.weight)
                if self.out_proj.bias is not None:
                    nn.init.zeros_(self.out_proj.bias)
            elif w_o is not None:
                if w_o.shape == self.out_proj.weight.shape:
                    self.out_proj.weight.copy_(w_o)
                elif w_o.t().contiguous().shape == self.out_proj.weight.shape:
                    self.out_proj.weight.copy_(w_o.t().contiguous())
                else:
                    self.out_proj.weight.zero_()
                    m_out = min(self.hidden_size, w_o.shape[0])
                    m_in = min(self.q_head_size, w_o.shape[1])
                    self.out_proj.weight[:m_out, :m_in].copy_(w_o[:m_out, :m_in])

                if b_o is not None and self.out_proj.bias is not None:
                    m_bo = min(self.out_proj.bias.shape[0], b_o.shape[0])
                    self.out_proj.bias[:m_bo].copy_(b_o[:m_bo])

                fnorm_o = torch.norm(self.out_proj.weight.float(), p="fro").item()
                if fnorm_o < 1e-4:
                    logger.warning(f"Copied out_proj weight has near-zero Frobenius norm (< 1e-4): {fnorm_o}")

            # Frobenius norm checks on instantiated projections
            for name, mod in [("q_proj", self.q_proj), ("k_proj", self.k_proj), ("v_proj", self.v_proj)]:
                if torch.norm(mod.weight.float(), p="fro").item() < 1e-4:
                    logger.warning(f"Copied weight {name} has near-zero Frobenius norm (< 1e-4)")

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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        # 1. Project Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(kv_input).view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(kv_input).view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # 2. RoPE support
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. GQA expansion
        if self.num_kv_heads < self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # 4. Attention scores in FP32
        scale = 1.0 / math.sqrt(self.head_dim)
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is None and head_mask is None and not output_attentions:
            try:
                _qs = q.half() if q.dtype == torch.float32 else q
                _ks = k.half() if k.dtype == torch.float32 else k
                _vs = v.half() if v.dtype == torch.float32 else v
                _is_causal = getattr(self, 'is_decoder', False) and not is_cross and seq_len > 1
                _sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                    _qs, _ks, _vs, is_causal=_is_causal, dropout_p=self.dropout.p if self.training else 0.0
                )
                _sdpa_out = _sdpa_out.to(hidden_states.dtype).transpose(1, 2).contiguous().view(batch_size, seq_len, self.all_head_size)
                _sdpa_out = torch.nan_to_num(_sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4)
                out = _sdpa_out if getattr(self, 'is_nested_encoder', False) else self.out_proj(_sdpa_out)
                if out.shape[-1] != hidden_states.shape[-1]:
                    if out.shape[-1] < hidden_states.shape[-1]:
                        out = F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
                    else:
                        out = out[..., :hidden_states.shape[-1]]
                return out, None
            except Exception:
                pass

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale

        # 5. Masking
        if self.is_decoder and not is_cross and seq_len > 1:
            causal_mask = torch.triu(
                torch.full((seq_len, kv_len), float("-inf"), device=scores.device, dtype=torch.float32),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
            scores = scores + causal_mask

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                scores = scores + (eff_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
            elif eff_mask.dim() == 3:
                scores = scores + eff_mask.unsqueeze(1).float()
            elif eff_mask.dim() == 4:
                scores = scores + eff_mask.float()

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        probs = F.softmax(scores, dim=-1).to(hidden_states.dtype)
        probs = self.dropout(probs)

        # 6. Safe head_mask application
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            probs = probs * head_mask

        # 7. Context computation & Head Merge
        context = torch.matmul(probs, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        # 8. Output projection (bypassed if nested encoder)
        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        if out.shape[-1] != hidden_states.shape[-1]:
            if out.shape[-1] < hidden_states.shape[-1]:
                out = F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
            else:
                out = out[..., :hidden_states.shape[-1]]

        return (out, probs if output_attentions else None)


@ATTENTION_REGISTRY.register("squeezing_heads_inject")
@ATTENTION_REGISTRY.register("squeezing_heads_distill_inject")
@ATTENTION_REGISTRY.register("squeezing_heads_distiller_inject")
@ATTENTION_REGISTRY.register("inject_squeezing_heads")
@ATTENTION_REGISTRY.register("inject_squeezing_heads_distill")
@ATTENTION_REGISTRY.register("inject_squeezing_heads_distiller")
def inject_squeezing_heads(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
):
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
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
    elif hasattr(target, "h") and isinstance(target.h, (torch.nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_squeezing_heads(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config if 'comp_config' in locals() or 'comp_config' in kwargs else None,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    if old_attn is None and layer is not None:
        old_attn = getattr(layer, "attention", getattr(layer, "attn", getattr(layer, "self_attn", None)))
        if old_attn is None and hasattr(layer, "self"):
            old_attn = layer
    if old_attn is None:
        return None

    config = cfg
    if config is None and hasattr(layer, "config"):
        config = layer.config
    if config is None and hasattr(old_attn, "config"):
        config = old_attn.config

    rank = kwargs.pop("rank", None)
    lora_rank = kwargs.pop("lora_rank", None)
    r = rank if rank is not None else _cfg_get(comp_config, "rank", _cfg_get(comp_config, "svd_rank", None))
    lr = lora_rank if lora_rank is not None else _cfg_get(comp_config, "lora_rank", 16)

    try:
        new_attn = SqueezingHeadsDistiller(
            config=config,
            old_attn=old_attn,
            lora_rank=lr,
            rank=r,
            **kwargs,
        )
    except Exception as e:
        logger.error("Failed to instantiate SqueezingHeadsDistiller at layer %d: %s", layer_idx, e)
        return None

    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new_attn
            new_attn.is_nested_encoder = True
        else:
            layer.attention = new_attn
    elif hasattr(layer, "attn"):
        layer.attn = new_attn
    elif hasattr(layer, "self_attn"):
        layer.self_attn = new_attn
    elif hasattr(layer, "self"):
        layer.self = new_attn
        new_attn.is_nested_encoder = True

    return new_attn


inject_squeezing_heads_distill = inject_squeezing_heads
inject_squeezing_heads_attention = inject_squeezing_heads
inject_squeezing_heads_distill_attention = inject_squeezing_heads
inject_squeezing_heads_distiller = inject_squeezing_heads
SqueezingHeads = SqueezingHeadsDistiller
SqueezingHeadsAttention = SqueezingHeadsDistiller
