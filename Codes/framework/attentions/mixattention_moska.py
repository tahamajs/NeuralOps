"""
MixAttention-MoSKA: Frequency/Variance-Aware Sparse Attention (FASA).
"""

import math
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("mixattention_moska")
# alias fasa removed
class MixAttentionMoSKA(BaseAttention):
    r"""
    MixAttention-MoSKA: Frequency/Variance-Aware Sparse Attention (FASA).
    Prunes key-value tokens based on variance importance while strictly protecting [CLS] (index 0).
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = 64,
        fasa_keep_ratio: float = 0.5,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.dim: int = getattr(config, "hidden_size", kwargs.get("hidden_size", 768))
        self.num_heads: int = getattr(config, "num_attention_heads", kwargs.get("num_attention_heads", 4))
        self.head_dim: int = self.dim // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size: int = self.num_heads * self.head_dim
        self.fasa_keep_ratio: float = fasa_keep_ratio

        self.W_q = nn.Linear(self.dim, self.all_head_size, bias=True)
        self.W_k = nn.Linear(self.dim, self.all_head_size, bias=True)
        self.W_v = nn.Linear(self.dim, self.all_head_size, bias=True)
        self.out_proj = nn.Linear(self.all_head_size, self.dim, bias=True)
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
                if q_src.bias is not None and self.W_q.bias is not None:
                    self.W_q.bias.copy_(q_src.bias)
            if k_src is not None and hasattr(k_src, "weight"):
                self.W_k.weight.copy_(k_src.weight)
                if k_src.bias is not None and self.W_k.bias is not None:
                    self.W_k.bias.copy_(k_src.bias)
            if v_src is not None and hasattr(v_src, "weight"):
                self.W_v.weight.copy_(v_src.weight)
                if v_src.bias is not None and self.W_v.bias is not None:
                    self.W_v.bias.copy_(v_src.bias)
            if out_src is not None and hasattr(out_src, "weight"):
                self.out_proj.weight.copy_(out_src.weight)
                if out_src.bias is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.copy_(out_src.bias)

    def _fasa_prune(
        self, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, H, L, D_h = k.shape
        if L <= 4:
            return k, v, mask

        importance = k.float().var(dim=-1).mean(dim=1)  # (B, L)
        if mask is not None:
            if mask.dim() == 2:
                mask_2d = mask.bool() if mask.dtype == torch.bool else mask > 0
                importance = importance.masked_fill(~mask_2d, -10000.0)
            elif mask.dim() == 4:
                mask_2d = mask[:, 0, 0, :] > -10000.0 if mask.min() < -1000 else mask[:, 0, 0, :] > 0
                importance = importance.masked_fill(~mask_2d, -10000.0)

        # Always protect [CLS] token at index 0 unconditionally
        importance[:, 0] = 1e9

        keep = max(2, min(L, int(L * self.fasa_keep_ratio)))
        _, idx = torch.topk(importance, keep, dim=-1)
        idx, _ = torch.sort(idx, dim=-1)

        idx_exp = idx.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D_h)
        k_pruned = k.gather(2, idx_exp)
        v_pruned = v.gather(2, idx_exp)

        pruned_mask = None
        if mask is not None:
            if mask.dim() == 2:
                pruned_mask = mask.gather(1, idx)
            elif mask.dim() == 4:
                mask_2d = mask[:, 0, 0, :] > -10000.0 if mask.min() < -1000 else mask[:, 0, 0, :] > 0
                pruned_mask = mask_2d.gather(1, idx)

        return k_pruned, v_pruned, pruned_mask

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
        mask = kwargs.get('mask', None)
        if attention_mask is None and mask is not None:
            attention_mask = mask

        x = hidden_states
        B, L, D = x.shape
        H = self.num_heads
        q = self.W_q(x).view(B, L, H, self.head_dim).permute(0, 2, 1, 3)
        k = self.W_k(x).view(B, L, H, self.head_dim).permute(0, 2, 1, 3)
        v = self.W_v(x).view(B, L, H, self.head_dim).permute(0, 2, 1, 3)

        k_p, v_p, p_mask = self._fasa_prune(k, v, attention_mask)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k_p.transpose(-2, -1)) * scale

        if p_mask is not None:
            pad_mask = (p_mask == 0).unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(pad_mask, -10000.0)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        out = torch.matmul(attn_weights, v_p).permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)
        output = self.out_proj(out)
        return output, (attn_weights if output_attentions else None)
