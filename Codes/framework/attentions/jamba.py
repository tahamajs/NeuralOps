"""
Jamba-2 Hybrid Transformer: SVD-Mamba + MHA + MoE Attention Block.
"""

from typing import Any, Optional, Tuple
import math
import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.simple_ssm import SimpleSSM
from framework.attentions.expert_choice_moe import ExpertChoiceMoE


class SVDMamba(nn.Module):
    def __init__(self, d_model: int = 768, config: Optional[Any] = None, rank: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__()
        if config is not None:
            d_model = getattr(config, "hidden_size", getattr(config, "d_model", d_model))
        self.d_model = d_model
        self.ssm = SimpleSSM(d_model=d_model)
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs: Any):
        x_proj = self.in_proj(hidden_states)
        ssm_out = self.ssm(x_proj, attention_mask=attention_mask)
        if isinstance(ssm_out, tuple):
            ssm_out = ssm_out[0]
        out = self.out_proj(ssm_out)
        return out, None


@ATTENTION_REGISTRY.register("jamba")
@ATTENTION_REGISTRY.register("jamba2")
class Jamba2Block(BaseAttention):
    r"""
    Jamba-2 Hybrid Architecture combining SVD-Mamba, Multi-Head Attention, and Expert Choice MoE.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        n_experts: int = 4,
        top_k: int = 2,
        rank: Optional[int] = 64,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        if config is not None:
            d_model = getattr(config, "hidden_size", getattr(config, "d_model", d_model or 768))
            n_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", n_heads or 12))
            n_experts = getattr(config, "num_experts", getattr(config, "n_experts", n_experts))
            top_k = getattr(config, "top_k", top_k)
        else:
            d_model = d_model or 768
            n_heads = n_heads or 12

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_out, self.kv_out, self.head_dim, self.kv_repeat = get_qkv_projection_dims(config)
        self.all_head_size = self.q_out

        self.q_lin = nn.Linear(d_model, self.all_head_size)
        self.k_lin = nn.Linear(d_model, getattr(self, "all_kv_head_size", self.all_head_size))
        self.v_lin = nn.Linear(d_model, getattr(self, "all_kv_head_size", self.all_head_size))
        self.out_lin = nn.Linear(self.all_head_size, d_model)

        self.mamba = SVDMamba(d_model, config=config, rank=rank)
        self.moe = ExpertChoiceMoE(d_model, n_experts=n_experts, capacity_factor=1.0)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        dropout_p = getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)) if config else 0.1
        self.dropout = nn.Dropout(dropout_p)
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        with torch.no_grad():
            q_src = getattr(pretrained_attn, "q_lin", getattr(pretrained_attn, "query", getattr(pretrained_attn, "q_proj", None)))
            k_src = getattr(pretrained_attn, "k_lin", getattr(pretrained_attn, "key", getattr(pretrained_attn, "k_proj", None)))
            v_src = getattr(pretrained_attn, "v_lin", getattr(pretrained_attn, "value", getattr(pretrained_attn, "v_proj", None)))
            out_src = getattr(pretrained_attn, "out_lin", getattr(pretrained_attn, "dense", getattr(pretrained_attn, "out_proj", getattr(pretrained_attn, "o_proj", None))))

            if q_src is not None and hasattr(q_src, "weight"):
                self.q_lin.weight.copy_(q_src.weight)
                if q_src.bias is not None:
                    self.q_lin.bias.copy_(q_src.bias)
            if k_src is not None and hasattr(k_src, "weight"):
                self.k_lin.weight.copy_(k_src.weight)
                if k_src.bias is not None:
                    self.k_lin.bias.copy_(k_src.bias)
            if v_src is not None and hasattr(v_src, "weight"):
                self.v_lin.weight.copy_(v_src.weight)
                if v_src.bias is not None:
                    self.v_lin.bias.copy_(v_src.bias)
            if out_src is not None and hasattr(out_src, "weight"):
                self.out_lin.weight.copy_(out_src.weight)
                if out_src.bias is not None:
                    self.out_lin.bias.copy_(out_src.bias)

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
        B, L, D = hidden_states.shape

        # Branch 1: SVD-Mamba
        mamba_out, _ = self.mamba(hidden_states, attention_mask=attention_mask)
        x1 = self.norm1(hidden_states + mamba_out)

        # Branch 2: Multi-Head Attention
        q = self.q_lin(x1).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_lin(x1).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_lin(x1).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(pad_mask, -10000.0)
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask.clamp(min=-10000.0).to(scores.dtype)

        probs = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs)
        if head_mask is not None:
            probs = probs * head_mask

        attn_out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, L, self.all_head_size)
        attn_proj = self.out_lin(attn_out)
        x2 = self.norm2(x1 + self.dropout(attn_proj))

        # Branch 3: Expert Choice MoE
        moe_out = self.moe(x2)
        x3 = self.norm3(x2 + self.dropout(moe_out))
        self.aux_loss = self.moe.aux_loss

        return x3, (probs if output_attentions else None)
