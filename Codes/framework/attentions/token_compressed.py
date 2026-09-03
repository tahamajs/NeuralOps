"""
Token-Compressed Attention.

Evaluates multi-head attention over sequence tokens dynamically compressed via top-k energy/norm
saliency scoring, strictly guaranteeing preservation of the [CLS] token (index 0).
"""

from typing import Any, Optional, Tuple
import math
import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("token_compressed")
@ATTENTION_REGISTRY.register("token_compressed_attention")
@ATTENTION_REGISTRY.register("token_compressed_module")
class TokenCompressedAttention(BaseAttention):
    r"""
    Token-Compressed Attention Module.
    Compresses Key and Value sequence tokens using energy/norm selection while preserving [CLS] (index 0).
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = 64,
        compression_ratio: float = 0.5,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.dim = getattr(config, "hidden_size", kwargs.get("hidden_size", 768))
        self.num_heads = getattr(config, "num_attention_heads", kwargs.get("num_attention_heads", 4))
        self.head_dim = self.dim // self.num_heads
        self.q_out, self.kv_out, self.head_dim, self.kv_repeat = get_qkv_projection_dims(config)
        self.all_head_size = self.q_out
        self.compression_ratio = compression_ratio

        self.q_lin = nn.Linear(self.dim, self.all_head_size)
        self.k_lin = nn.Linear(self.dim, getattr(self, "all_kv_head_size", self.all_head_size))
        self.v_lin = nn.Linear(self.dim, getattr(self, "all_kv_head_size", self.all_head_size))
        self.out_lin = nn.Linear(self.all_head_size, self.dim)

        self.score_net = nn.Sequential(
            nn.Linear(self.dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
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

    def _select_compressed_tokens(
        self, x: torch.Tensor, k_tensor: torch.Tensor, v_tensor: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        if L <= 4:
            return k_tensor, v_tensor, mask

        # Saliency scoring based on token norm + learnable score
        energy = torch.norm(x, p=2, dim=-1)  # (B, L)
        learned_score = self.score_net(x).squeeze(-1)  # (B, L)
        total_importance = energy + learned_score

        k_keep = max(2, min(L, int(L * self.compression_ratio)))

        # Token 0 is always [CLS] - give it highest importance
        total_importance[:, 0] = 1e4

        if mask is not None:
            if mask.dim() == 2:
                total_importance = total_importance.masked_fill(mask == 0, -10000.0)
                total_importance[:, 0] = 1e4

        _, top_indices = torch.topk(total_importance, k_keep, dim=-1)  # (B, k_keep)
        # Sort indices to preserve sequence ordering
        top_indices, _ = torch.sort(top_indices, dim=-1)

        # Gather K and V: [B, H, k_keep, Dh]
        top_idx_exp = top_indices.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, Dh)
        k_comp = torch.gather(k_tensor, 2, top_idx_exp)
        v_comp = torch.gather(v_tensor, 2, top_idx_exp)

        comp_mask = None
        if mask is not None and mask.dim() == 2:
            comp_mask = torch.gather(mask, 1, top_indices)

        return k_comp, v_comp, comp_mask

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

        q = self.q_lin(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k  = self.k_lin(hidden_states).view(B, L, getattr(self, "num_kv_heads", self.num_heads), self.head_dim).transpose(1, 2)
        k = repeat_kv(k, getattr(self, 'kv_repeat', 1))
        v  = self.v_lin(hidden_states).view(B, L, getattr(self, "num_kv_heads", self.num_heads), self.head_dim).transpose(1, 2)
        v = repeat_kv(v, getattr(self, 'kv_repeat', 1))

        # Compress KV tokens while preserving [CLS]
        k_comp, v_comp, comp_mask = self._select_compressed_tokens(hidden_states, k, v, attention_mask)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k_comp.transpose(-2, -1)) * scale  # (B, H, L, k_keep)

        if comp_mask is not None:
            pad_mask = (comp_mask == 0).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask, -10000.0)

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        if head_mask is not None:
            probs = probs * head_mask

        attn_out = torch.matmul(probs, v_comp).transpose(1, 2).contiguous().view(B, L, self.all_head_size)
        out = self.out_lin(attn_out)

        return out, (probs if output_attentions else None)
