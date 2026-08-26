"""
HymbaTransX: Hybrid Differential Transformer & State Space Model Block.
"""

import math
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.simple_ssm import SimpleSSM


class HeadRoPE(nn.Module):
    r"""
    Head-wise Rotary Position Embedding (RoPE) applied to Q and K.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(1), persistent=False)  # (1, 1, seq_len, head_dim)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, L, d]
        B, H, L, d = x.shape
        if L > self.max_seq_len:
            self._build_cache(L)
            self.max_seq_len = L

        cos = self.cos_cached[:, :, :L, :d]
        sin = self.sin_cached[:, :, :L, :d]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even = x_even * cos[..., 0::2] - x_odd * sin[..., 0::2]
        rot_odd = x_odd * cos[..., 0::2] + x_even * sin[..., 0::2]
        return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)


@ATTENTION_REGISTRY.register("hymba_transx")
class HymbaTransXBlock(BaseAttention):
    r"""
    HymbaTransX: Hybrid Differential Transformer & State Space Model Block.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = 64,
        num_heads: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.dim: int = getattr(config, "hidden_size", kwargs.get("hidden_size", 768))
        self.num_heads: int = num_heads or getattr(config, "num_attention_heads", kwargs.get("num_attention_heads", 12))
        self.head_dim: int = self.dim // self.num_heads if self.num_heads > 0 else 64

        self.rope = HeadRoPE(self.head_dim)
        self.W_q1 = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k1 = nn.Linear(self.dim, self.dim, bias=False)
        self.W_q2 = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k2 = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)
        self.mamba = SimpleSSM(d_model=self.dim, config=config)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))
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
                self.W_q1.weight.copy_(q_src.weight)
                self.W_q2.weight.copy_(q_src.weight.data * 0.9 if hasattr(q_src.weight, "data") else q_src.weight)
            if k_src is not None and hasattr(k_src, "weight"):
                self.W_k1.weight.copy_(k_src.weight)
                self.W_k2.weight.copy_(k_src.weight.data * 0.9 if hasattr(k_src.weight, "data") else k_src.weight)
            if v_src is not None and hasattr(v_src, "weight"):
                self.W_v.weight.copy_(v_src.weight)
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
        x = hidden_states
        B, L, D = x.shape

        q1 = self.rope(self.W_q1(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2))
        k1 = self.rope(self.W_k1(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2))
        q2 = self.rope(self.W_q2(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2))
        k2 = self.rope(self.W_k2(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2))
        v = self.W_v(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        scores2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
                scores1 = scores1.masked_fill(pad_mask, -10000.0)
                scores2 = scores2.masked_fill(pad_mask, -10000.0)
            elif attention_mask.dim() == 4:
                scores1 = scores1 + attention_mask
                scores2 = scores2 + attention_mask

        attn1 = F.softmax(scores1, dim=-1)
        attn2 = F.softmax(scores2, dim=-1)
        diff_attn = F.relu(attn1 - 0.2 * attn2)
        diff_attn = diff_attn / (diff_attn.sum(dim=-1, keepdim=True) + 1e-8)
        diff_attn = self.dropout(diff_attn)

        if head_mask is not None:
            diff_attn = diff_attn * head_mask

        A_out = torch.matmul(diff_attn, v).transpose(1, 2).contiguous().view(B, L, D)
        mamba_res = self.mamba(x, attention_mask=attention_mask)
        M_out = mamba_res[0] if isinstance(mamba_res, tuple) else mamba_res

        alpha = torch.sigmoid(self.alpha)
        out = self.out_proj(alpha * A_out + (1.0 - alpha) * M_out)

        return out, (diff_attn if output_attentions else None)
