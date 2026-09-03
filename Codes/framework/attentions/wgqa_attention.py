"""
Weighted Grouped-Query Attention (WGQA).
"""

import math
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("wgqa", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("wgqa_attention")
class WeightedGroupedQueryAttention(BaseAttention):
    r"""
    Weighted Grouped-Query Attention with learnable per-channel KV importance weights
    and head-pooling pretrained weight transfer.
    """

    supports_gqa: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        num_kv_heads: int = 4,
        rank: Optional[int] = 64,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", getattr(config, "num_heads", kwargs.get("num_attention_heads", 4)))
        self.hidden_size: int = getattr(config, "hidden_size", kwargs.get("hidden_size", 768))
        self.head_dim: int = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        cfg_kv = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None))
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        else:
            self.num_kv_heads = min(num_kv_heads, self.num_heads)
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))

        # Learnable per-KV-head importance weights
        self.k_weights = nn.Parameter(torch.ones(self.num_kv_heads, 1))
        self.v_weights = nn.Parameter(torch.ones(self.num_kv_heads, 1))

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

        dropout_p = getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)) if config else 0.1
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        with torch.no_grad():
            q_src = getattr(pretrained_attn, "q_lin", getattr(pretrained_attn, "query", getattr(pretrained_attn, "q_proj", None)))
            k_src = getattr(pretrained_attn, "k_lin", getattr(pretrained_attn, "key", getattr(pretrained_attn, "k_proj", None)))
            v_src = getattr(pretrained_attn, "v_lin", getattr(pretrained_attn, "value", getattr(pretrained_attn, "v_proj", None)))
            out_src = getattr(pretrained_attn, "out_lin", getattr(pretrained_attn, "dense", getattr(pretrained_attn, "out_proj", getattr(pretrained_attn, "o_proj", None))))

            if q_src is not None and hasattr(q_src, "weight"):
                self.q_proj.weight.copy_(q_src.weight)
                if q_src.bias is not None:
                    self.q_proj.bias.copy_(q_src.bias)

            if k_src is not None and hasattr(k_src, "weight"):
                # Pool source heads into KV heads
                rep = self.num_heads // self.num_kv_heads
                w_k = k_src.weight.view(self.num_kv_heads, rep, self.head_dim, self.hidden_size).mean(dim=1).view(-1, self.hidden_size)
                self.k_proj.weight.copy_(w_k)
                if k_src.bias is not None:
                    b_k = k_src.bias.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
                    self.k_proj.bias.copy_(b_k)

            if v_src is not None and hasattr(v_src, "weight"):
                rep = self.num_heads // self.num_kv_heads
                w_v = v_src.weight.view(self.num_kv_heads, rep, self.head_dim, self.hidden_size).mean(dim=1).view(-1, self.hidden_size)
                self.v_proj.weight.copy_(w_v)
                if v_src.bias is not None:
                    b_v = v_src.bias.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
                    self.v_proj.bias.copy_(b_v)

            if out_src is not None and hasattr(out_src, "weight"):
                self.out_proj.weight.copy_(out_src.weight)
                if out_src.bias is not None:
                    self.out_proj.bias.copy_(out_src.bias)

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
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        k = k * self.k_weights.view(1, 1, self.num_kv_heads, 1)
        v = v * self.v_weights.view(1, 1, self.num_kv_heads, 1)

        repeat_factor = max(1, self.num_heads // max(1, self.num_kv_heads))
        if repeat_factor > 1:
            k = k.repeat_interleave(repeat_factor, dim=2)
            v = v.repeat_interleave(repeat_factor, dim=2)
        if k.shape[2] < self.num_heads:
            diff = self.num_heads - k.shape[2]
            k = torch.cat([k, k[:, :, :diff]], dim=2)
            v = torch.cat([v, v[:, :, :diff]], dim=2)
        elif k.shape[2] > self.num_heads:
            k = k[:, :, :self.num_heads]
            v = v[:, :, :self.num_heads]

        q = q.transpose(1, 2)  # (B, H_q, L, d)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                scores = scores + (attention_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask

        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        context = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(context)

        return out, (attn_probs if output_attentions else None)
