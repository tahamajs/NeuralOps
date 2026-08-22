"""
Block Transformer with Hierarchical Intra-Block and Inter-Block Attention (HIBA).
Fixes:
- Non-causal bidirectional intra-block attention for encoder architectures.
- Guaranteed CLS token block inclusion in inter-block sparse routing.
- Pretrained weight initialization via copy_weights.
"""

import math
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("block_transformer_hiba")
class BlockTransformerHIBA(BaseAttention):
    def __init__(
        self,
        config: Optional[Any] = None,
        rank: int = 64,
        block_size: int = 64,
        top_k_inter: int = 4,
        rectification_interval: int = 4,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.dim = getattr(config, "hidden_size", kwargs.get("hidden_size", 768))
        self.num_heads = getattr(config, "num_attention_heads", kwargs.get("num_attention_heads", 12))
        self.head_dim = self.dim // self.num_heads
        self.block_size = block_size
        self.top_k_inter = top_k_inter
        self.rectification_interval = rectification_interval
        self.is_causal = getattr(config, "is_causal", kwargs.get("is_causal", False))

        self.W_q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim)

        self.layer_idx = 0
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
                self.W_q.weight.copy_(q_src.weight)
            if k_src is not None and hasattr(k_src, "weight"):
                self.W_k.weight.copy_(k_src.weight)
            if v_src is not None and hasattr(v_src, "weight"):
                self.W_v.weight.copy_(v_src.weight)
            if out_src is not None and hasattr(out_src, "weight"):
                self.out_proj.weight.copy_(out_src.weight)
                if out_src.bias is not None:
                    self.out_proj.bias.copy_(out_src.bias)

    def _convert_mask_to_bool(self, attention_mask, B, L, device):
        if attention_mask is None:
            return torch.ones(B, L, device=device, dtype=torch.bool)
        if attention_mask.dim() == 2:
            return attention_mask.bool() if attention_mask.dtype == torch.bool else attention_mask > 0
        elif attention_mask.dim() == 4:
            return attention_mask[:, 0, 0, :] > -10000.0 if attention_mask.min() < -1000 else attention_mask[:, 0, 0, :] > 0
        return torch.ones(B, L, device=device, dtype=torch.bool)

    def _intra_block_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        B, H, L_b, _ = q.shape
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if causal:
            causal_mask = torch.tril(torch.ones(L_b, L_b, device=q.device)).bool()
            attn = attn.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -10000.0)

        pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        attn = attn.masked_fill(pad_mask, -10000.0)

        probs = F.softmax(attn, dim=-1)
        probs = self.dropout(probs)
        return torch.matmul(probs, v)

    def _inter_block_sparse(
        self,
        q_b: torch.Tensor,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        block_idx: int,
        num_blocks: int,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, H, L_b, _ = q_b.shape
        total_len = k_all.size(2)
        start_b = block_idx * self.block_size
        end_b = min((block_idx + 1) * self.block_size, total_len)

        k_select = []
        v_select = []
        valid_select = []

        # Always include block 0 (which contains CLS token at index 0)
        if block_idx != 0:
            end_0 = min(self.block_size, total_len)
            k_select.append(k_all[:, :, :end_0, :])
            v_select.append(v_all[:, :, :end_0, :])
            valid_select.append(valid_mask[:, :end_0])

        if block_idx > 0 and block_idx != 1:
            k_select.append(k_all[:, :, :start_b, :])
            v_select.append(v_all[:, :, :start_b, :])
            valid_select.append(valid_mask[:, :start_b])

        if (block_idx + 1) * self.block_size < total_len:
            k_select.append(k_all[:, :, end_b:, :])
            v_select.append(v_all[:, :, end_b:, :])
            valid_select.append(valid_mask[:, end_b:])

        if not k_select:
            return torch.zeros_like(q_b)

        k_neighbors = torch.cat(k_select, dim=2)
        v_neighbors = torch.cat(v_select, dim=2)
        valid_neighbors = torch.cat(valid_select, dim=1)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q_b, k_neighbors.transpose(-2, -1)) * scale

        pad_mask = ~valid_neighbors.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(pad_mask, -10000.0)

        top_k = min(self.top_k_inter, k_neighbors.size(2))
        _, topk_idx = torch.topk(scores, top_k, dim=-1)
        mask = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)
        scores = scores.masked_fill(mask == 0, -10000.0)

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        return torch.matmul(probs, v_neighbors)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = hidden_states.shape
        H = self.num_heads

        valid_mask = self._convert_mask_to_bool(attention_mask, B, L, hidden_states.device)

        q = self.W_q(hidden_states).view(B, L, H, self.head_dim).permute(0, 2, 1, 3)
        k = self.W_k(hidden_states).view(B, L, H, self.head_dim).permute(0, 2, 1, 3)
        v = self.W_v(hidden_states).view(B, L, H, self.head_dim).permute(0, 2, 1, 3)

        num_blocks = (L + self.block_size - 1) // self.block_size
        use_rectify = (self.layer_idx % self.rectification_interval == 0) or (num_blocks <= 2)
        self.layer_idx += 1

        if use_rectify:
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            pad_mask = ~valid_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask, -10000.0)
            attn_probs = F.softmax(scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            out = torch.matmul(attn_probs, v)
            return_attn = attn_probs if output_attentions else None
        else:
            out_list = []
            for b_idx in range(num_blocks):
                start = b_idx * self.block_size
                end = min(start + self.block_size, L)
                q_b = q[:, :, start:end, :]
                k_b = k[:, :, start:end, :]
                v_b = v[:, :, start:end, :]
                block_valid = valid_mask[:, start:end]
                block_pad_mask = ~block_valid

                intra_out = self._intra_block_attention(q_b, k_b, v_b, block_pad_mask, causal=self.is_causal)
                inter_out = self._inter_block_sparse(q_b, k, v, b_idx, num_blocks, valid_mask)
                out_b = intra_out + inter_out
                out_list.append(out_b)
            out = torch.cat(out_list, dim=2)
            return_attn = None

        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out), return_attn
