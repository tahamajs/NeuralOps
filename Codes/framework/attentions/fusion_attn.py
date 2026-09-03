"""
FUSIONATTN: Multi-Paradigm Hybrid Sparse Attention Module.
"""

import math
from typing import Any, Optional, Tuple
import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("fusion_attn")
class FUSIONATTN(BaseAttention):
    r"""
    FUSIONATTN: Hybrid Sparse Attention with Block Gating and Subspace Estimation.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        hidden_size: int = 768,
        num_heads = 4,
        block_size: int = 64,
        top_k_blocks: int = 4,
        rank: Optional[int] = 64,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        if config is not None:
            hidden_size = getattr(config, "hidden_size", hidden_size)
            num_heads = getattr(config, "num_attention_heads", num_heads)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_gate = nn.Sequential(
            nn.Linear(self.head_dim, 4 * self.head_dim),
            nn.GELU(),
            nn.Linear(4 * self.head_dim, 1),
        )
        self.gate_temp = nn.Parameter(torch.ones(1) * 1.0)
        dropout_p = getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)) if config else 0.1
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        with torch.no_grad():
            # Try to detect fused c_attn (GPT-2 style): one module holds Q+K+V concatenated
            c_attn = getattr(pretrained_attn, "c_attn", None)
            c_proj = getattr(pretrained_attn, "c_proj", None)
            if c_attn is not None and hasattr(c_attn, "weight"):
                # GPT-2 Conv1D: weight is [in, 3*out], so after transpose [3*out, in]
                W = c_attn.weight.data  # shape [in_features, 3*n_embd] for Conv1D
                if W.shape[0] < W.shape[1]:
                    W = W.t().contiguous()  # now [3*n_embd, n_embd]
                B_bias = getattr(c_attn, "bias", None)
                n_embd = W.shape[0] // 3
                wq = W[:n_embd]; wk = W[n_embd:2*n_embd]; wv = W[2*n_embd:]
                bq = B_bias[:n_embd] if B_bias is not None else None
                bk = B_bias[n_embd:2*n_embd] if B_bias is not None else None
                bv = B_bias[2*n_embd:] if B_bias is not None else None
                self.q_proj.weight.data.copy_(wq)
                self.k_proj.weight.data.copy_(wk)
                self.v_proj.weight.data.copy_(wv)
                if bq is not None and self.q_proj.bias is not None:
                    self.q_proj.bias.data.copy_(bq)
                if bk is not None and self.k_proj.bias is not None:
                    self.k_proj.bias.data.copy_(bk)
                if bv is not None and self.v_proj.bias is not None:
                    self.v_proj.bias.data.copy_(bv)
            else:
                q_src = getattr(pretrained_attn, "q_lin", getattr(pretrained_attn, "query", getattr(pretrained_attn, "q_proj", None)))
                k_src = getattr(pretrained_attn, "k_lin", getattr(pretrained_attn, "key", getattr(pretrained_attn, "k_proj", None)))
                v_src = getattr(pretrained_attn, "v_lin", getattr(pretrained_attn, "value", getattr(pretrained_attn, "v_proj", None)))
                if q_src is not None and hasattr(q_src, "weight"):
                    self.q_proj.weight.data.copy_(q_src.weight.data)
                    if q_src.bias is not None and self.q_proj.bias is not None:
                        self.q_proj.bias.data.copy_(q_src.bias.data)
                if k_src is not None and hasattr(k_src, "weight"):
                    self.k_proj.weight.data.copy_(k_src.weight.data)
                    if k_src.bias is not None and self.k_proj.bias is not None:
                        self.k_proj.bias.data.copy_(k_src.bias.data)
                if v_src is not None and hasattr(v_src, "weight"):
                    self.v_proj.weight.data.copy_(v_src.weight.data)
                    if v_src.bias is not None and self.v_proj.bias is not None:
                        self.v_proj.bias.data.copy_(v_src.bias.data)

            # Output projection — try c_proj first (GPT-2), then common aliases
            out_src = c_proj if c_proj is not None else getattr(
                pretrained_attn, "out_lin", getattr(pretrained_attn, "dense",
                getattr(pretrained_attn, "out_proj", getattr(pretrained_attn, "o_proj", None))))
            if out_src is not None and hasattr(out_src, "weight"):
                W_out = out_src.weight.data
                if W_out.shape[0] < W_out.shape[1]:  # Conv1D transposed
                    W_out = W_out.t().contiguous()
                self.out_proj.weight.data.copy_(W_out)
                if out_src.bias is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.data.copy_(out_src.bias.data)

    def _moba_block_attention(self, q, k, v, gate_scores):
        B, H, N, D = q.shape
        pad_len = (self.block_size - (N % self.block_size)) % self.block_size
        if pad_len > 0:
            q_pad = F.pad(q, (0, 0, 0, pad_len))
            k_pad = F.pad(k, (0, 0, 0, pad_len))
            v_pad = F.pad(v, (0, 0, 0, pad_len))
        else:
            q_pad, k_pad, v_pad = q, k, v

        num_blocks = q_pad.size(2) // self.block_size
        top_k = min(self.top_k_blocks, num_blocks)
        _, top_indices = torch.topk(gate_scores, top_k, dim=-1)
        # Always retain block 0 (which contains CLS token at index 0)
        top_indices[:, :, :, 0] = 0

        q_blocks = q_pad.reshape(B, H, num_blocks, self.block_size, D)
        k_blocks = k_pad.reshape(B, H, num_blocks, self.block_size, D)
        v_blocks = v_pad.reshape(B, H, num_blocks, self.block_size, D)

        out_blocks = []
        for b in range(num_blocks):
            idx_b = top_indices[:, :, b, :]
            idx_exp = idx_b.unsqueeze(-1).unsqueeze(-1).expand(B, H, top_k, self.block_size, D)
            k_sel = torch.gather(k_blocks, 2, idx_exp).reshape(B, H, top_k * self.block_size, D)
            v_sel = torch.gather(v_blocks, 2, idx_exp).reshape(B, H, top_k * self.block_size, D)

            q_curr = q_blocks[:, :, b, :, :]
            attn_b = torch.matmul(q_curr, k_sel.transpose(-2, -1)) / (D**0.5)
            attn_b = F.softmax(attn_b, dim=-1)
            attn_b = self.dropout(attn_b)
            out_b = torch.matmul(attn_b, v_sel)
            out_blocks.append(out_b)

        out_pad = torch.cat(out_blocks, dim=2)
        return out_pad[:, :, :N, :]

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
        B, N, D = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k   = self.k_proj(hidden_states).view(B, N, getattr(self, "num_kv_heads", getattr(self, "num_kv_heads", self.num_heads)), self.head_dim).transpose(1, 2)
        k = repeat_kv(k, getattr(self, 'kv_repeat', 1))
        k = repeat_kv(k, getattr(self, 'kv_repeat', 1))
        v   = self.v_proj(hidden_states).view(B, N, getattr(self, "num_kv_heads", getattr(self, "num_kv_heads", self.num_heads)), self.head_dim).transpose(1, 2)
        v = repeat_kv(v, getattr(self, 'kv_repeat', 1))
        v = repeat_kv(v, getattr(self, 'kv_repeat', 1))

        if N <= 128:
            # Full attention for short sequences
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    pad_mask = (attention_mask == 0).to(scores.device).unsqueeze(1).unsqueeze(2)
                    scores = scores.masked_fill(pad_mask, -10000.0)
                elif attention_mask.dim() == 4:
                    scores = scores + attention_mask.to(device=scores.device, dtype=scores.dtype)

            probs = F.softmax(scores, dim=-1)
            probs = self.dropout(probs)
            if head_mask is not None:
                probs = probs * head_mask

            context = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, N, D)
            return self.out_proj(context), (probs if output_attentions else None)

        pad_len = (self.block_size - (N % self.block_size)) % self.block_size
        q_padded = F.pad(q, (0, 0, 0, pad_len)) if pad_len > 0 else q
        k_padded = F.pad(k, (0, 0, 0, pad_len)) if pad_len > 0 else k
        num_blocks = q_padded.size(2) // self.block_size
        q_blocks = q_padded.reshape(B, self.num_heads, num_blocks, self.block_size, self.head_dim).mean(dim=-2)
        k_blocks  = k_padded.reshape(B, getattr(self, "num_kv_heads", self.num_heads), num_blocks, self.block_size, self.head_dim).mean(dim=-2)
        k_blocks = repeat_kv(k_blocks, getattr(self, 'kv_repeat', 1))

        q_gate = self.attn_gate(q_blocks).squeeze(-1)
        k_gate = self.attn_gate(k_blocks).squeeze(-1)
        gate_scores = torch.matmul(q_gate.unsqueeze(-1), k_gate.unsqueeze(-2)) / (self.gate_temp.abs() + 1e-5)

        context = self._moba_block_attention(q, k, v, gate_scores)
        context = context.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(context), None
