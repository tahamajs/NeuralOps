"""
SAMBA-M3: Hybrid Sliding-Window Attention and Spectral Mamba Block.
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
from framework.core.attention_mask import apply_attention_mask
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.simple_ssm import SimpleSSM

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) with partial dimension support."""
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    while cos.dim() < q_rot.dim():
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot
    k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot
    return q, k


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


class SpectralMamba(nn.Module):
    r"""Mamba with learnable spectral gating."""

    def __init__(self, d_model: int = 768, config: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__()
        if config is not None:
            d_model = getattr(config, "hidden_size", getattr(config, "d_model", d_model))
        elif hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        self.mamba = SimpleSSM(config=config, d_model=d_model, **kwargs)
        self.dt_scale = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = self.mamba(x, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        gate = torch.sigmoid(self.dt_scale).unsqueeze(0).unsqueeze(0)
        out = out * gate
        return out, None


@ATTENTION_REGISTRY.register("samba_m3", supports_causal=True, supports_gqa=True)
class SAMBAM3Block(nn.Module):
    r"""SAMBA-M3: Combination of Local Attention and Spectral Mamba."""

    supports_gqa: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        window: int = 128,
        rank: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if isinstance(config, int):
            d_model = config
            config = None
        if d_model is None:
            if config is not None:
                d_model = getattr(config, "hidden_size", getattr(config, "d_model", 768))
            else:
                d_model = kwargs.get("hidden_size", 768)
        if n_heads is None:
            n_heads = getattr(
                config,
                "num_attention_heads",
                getattr(config, "n_heads", kwargs.get("num_attention_heads", 12)),
            )
        window = getattr(config, "window_size", window)

        self.d_model: int = int(d_model)
        self.n_heads: int = int(n_heads)
        self.head_dim: int = self.d_model // self.n_heads if self.n_heads > 0 else 64
        self.window: int = window
        self.lora_rank: int = lora_rank
        self.lora_alpha: float = lora_alpha

        cfg_kv = (
            getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None))
            if config is not None
            else None
        )
        self.num_kv_heads: int = int(cfg_kv) if cfg_kv is not None else self.n_heads
        self.num_key_value_groups: int = max(1, self.n_heads // max(1, self.num_kv_heads))

        self.q_head_size: int = self.n_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        # Decoder / Causal flag
        if is_decoder is not None:
            self.is_decoder: bool = is_decoder
        elif config is not None:
            self.is_decoder = bool(getattr(config, "is_decoder", False))
        else:
            self.is_decoder = False

        self.is_nested_encoder: bool = False

        self.q_lin = LoRALinear(
            self.d_model,
            self.q_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.k_lin = LoRALinear(
            self.d_model,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.v_lin = LoRALinear(
            self.d_model,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.out_lin = LoRALinear(
            self.q_head_size,
            self.d_model,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        self.mamba = SpectralMamba(self.d_model, config=config, is_decoder=self.is_decoder, **kwargs)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(self.d_model)
        dropout_p = (
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        """Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA."""
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            src_cls_name = getattr(pretrained_attn, "__class__", type(pretrained_attn)).__name__
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(pretrained_attn, "query")
                and not hasattr(pretrained_attn, "dense")
                and not hasattr(pretrained_attn, "out_proj")
            ):
                self.is_nested_encoder = True

            if hasattr(pretrained_attn, "self") and hasattr(pretrained_attn, "output"):
                self.is_nested_encoder = True
                src = pretrained_attn.self
                if hasattr(pretrained_attn.output, "dense"):
                    w_o = pretrained_attn.output.dense.weight
                    b_o = pretrained_attn.output.dense.bias
            else:
                src = pretrained_attn

            # Pattern 1: GPT-2 Fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach()
                c_attn_b = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                if c_attn_w.shape[0] == self.d_model and c_attn_w.shape[1] != self.d_model:
                    c_attn_w = c_attn_w.t()

                q_dim = self.q_head_size
                k_dim = self.kv_head_size
                v_dim = self.kv_head_size

                w_q = c_attn_w[:q_dim, :]
                w_k = c_attn_w[q_dim : q_dim + k_dim, :]
                w_v = c_attn_w[q_dim + k_dim : q_dim + k_dim + v_dim, :]

                if c_attn_b is not None:
                    b_q = c_attn_b[:q_dim]
                    b_k = c_attn_b[q_dim : q_dim + k_dim]
                    b_v = c_attn_b[q_dim + k_dim : q_dim + k_dim + v_dim]

                out_w = src.c_proj.weight.detach()
                if out_w.shape[0] == self.d_model and out_w.shape[1] != self.d_model:
                    out_w = out_w.t()
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

            # Pattern 5: UniversalAdapter fallback
            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter()
                    w_q, b_q = adapter.get_weight_bias(src, "q")
                    w_k, b_k = adapter.get_weight_bias(src, "k")
                    w_v, b_v = adapter.get_weight_bias(src, "v")
                    if w_o is None:
                        w_o, b_o = adapter.get_weight_bias(src, "o")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            # Strict validation
            missing = []
            if w_q is None:
                missing.append("Q projection")
            if w_k is None:
                missing.append("K projection")
            if w_v is None:
                missing.append("V projection")
            if w_o is None and not self.is_nested_encoder:
                missing.append("Out projection")

            if missing:
                raise RuntimeError(
                    f"SAMBAM3Block: Failed to extract weights: {', '.join(missing)} from {type(pretrained_attn)}"
                )

            self.q_lin.weight.copy_(w_q.to(self.q_lin.weight.dtype))
            if b_q is not None and self.q_lin.bias is not None:
                self.q_lin.bias.copy_(b_q.to(self.q_lin.bias.dtype))
            elif self.q_lin.bias is not None:
                self.q_lin.bias.zero_()

            self.k_lin.weight.copy_(w_k.to(self.k_lin.weight.dtype))
            if b_k is not None and self.k_lin.bias is not None:
                self.k_lin.bias.copy_(b_k.to(self.k_lin.bias.dtype))
            elif self.k_lin.bias is not None:
                self.k_lin.bias.zero_()

            self.v_lin.weight.copy_(w_v.to(self.v_lin.weight.dtype))
            if b_v is not None and self.v_lin.bias is not None:
                self.v_lin.bias.copy_(b_v.to(self.v_lin.bias.dtype))
            elif self.v_lin.bias is not None:
                self.v_lin.bias.zero_()

            if w_o is not None:
                self.out_lin.weight.copy_(w_o.to(self.out_lin.weight.dtype))
                if b_o is not None and self.out_lin.bias is not None:
                    self.out_lin.bias.copy_(b_o.to(self.out_lin.bias.dtype))
                elif self.out_lin.bias is not None:
                    self.out_lin.bias.zero_()

            f_norm_q = float(torch.linalg.norm(self.q_lin.weight))
            if f_norm_q < 1e-4:
                raise RuntimeError(f"SAMBAM3Block: Q weight norm is near zero: {f_norm_q}")

            # Also transfer weights to spectral mamba
            try:
                self.mamba.mamba.copy_weights(pretrained_attn)
            except Exception as e:
                logger.debug("Failed copying weights to mamba: %s", e)

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
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        B, L, D = hidden_states.shape

        q = self.q_lin(hidden_states).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_lin(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_lin(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Partial Rotary Position Embeddings (RoPE)
        pos_emb = kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, tuple) and len(pos_emb) == 2:
            cos, sin = pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Grouped-Query Attention expansion
        if self.num_kv_heads != self.n_heads:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

        # Sliding window mask
        idx = torch.arange(L, device=q.device)
        if self.is_decoder:
            diff = idx.unsqueeze(0) - idx.unsqueeze(1)  # [L, L], diff[i, j] = j - i
            causal_window_mask = (diff > 0) | ((-diff) > self.window)
            scores = scores.masked_fill(causal_window_mask.unsqueeze(0).unsqueeze(0), -10000.0)
        else:
            dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
            window_mask = dist > self.window
            scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), -10000.0)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(pad_mask, -10000.0)
            elif attention_mask.dim() == 3:
                scores = scores + attention_mask.unsqueeze(1).to(scores.dtype)
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask.to(scores.dtype)

        probs = F.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs.to(v.dtype))
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
            probs = probs * head_mask

        attn_out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, L, D)

        mamba_out, _ = self.mamba(hidden_states, attention_mask=attention_mask, **kwargs)
        alpha = torch.sigmoid(self.alpha)

        if self.is_nested_encoder:
            out = alpha * attn_out + (1.0 - alpha) * mamba_out
        else:
            attn_proj = self.out_lin(attn_out)
            out = self.norm(hidden_states + alpha * attn_proj + (1.0 - alpha) * mamba_out)

        return (out, (probs if output_attentions else None))
