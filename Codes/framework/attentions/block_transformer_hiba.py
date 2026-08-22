"""
Block Transformer with Hierarchical Intra-Block and Inter-Block Attention (HIBA).
Fixes:
- Apply padding mask in both intra and inter block attention.
- Consistent mask handling (bool/0-1/additive).
- output_attentions respected (returns None for sparse path).
"""
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
        config: Any,
        rank: int = 64,
        block_size: int = 512,
        top_k_inter: int = 32,
        rectification_interval: int = 4,
    ) -> None:
        super().__init__(config, rank)
        self.dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dim // self.num_heads
        self.block_size = block_size
        self.top_k_inter = top_k_inter
        self.rectification_interval = rectification_interval

        self.W_q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim)

        self.layer_idx = 0

    def _convert_mask_to_bool(self, attention_mask, B, L, device):
        """Return boolean valid mask [B, L] (True for valid, False for padding)."""
        if attention_mask is None:
            return torch.ones(B, L, device=device, dtype=torch.bool)
        if attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                return attention_mask
            else:
                return attention_mask > 0
        elif attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                return attention_mask[:, 0, 0, :]
            else:
                if attention_mask.min() < -1000:
                    return attention_mask[:, 0, 0, :] > -10000.0
                else:
                    return attention_mask[:, 0, 0, :] > 0
        else:
            raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

    def _intra_block_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor,  # [B, L_b] True for pad
        causal: bool = True,
    ) -> torch.Tensor:
        B, H, L_b, _ = q.shape
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if causal:
            causal_mask = torch.tril(torch.ones(L_b, L_b, device=q.device)).bool()
            attn = attn.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Apply padding mask (mask out keys that are padding)
        pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L_b]
        attn = attn.masked_fill(pad_mask, float("-inf"))

        probs = F.softmax(attn, dim=-1)
        return torch.matmul(probs, v)

    def _inter_block_sparse(
        self,
        q_b: torch.Tensor,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        block_idx: int,
        num_blocks: int,
        valid_mask: torch.Tensor,  # [B, L]
    ) -> torch.Tensor:
        B, H, L_b, _ = q_b.shape
        total_len = k_all.size(2)
        start_b = block_idx * self.block_size
        end_b = min((block_idx + 1) * self.block_size, total_len)

        # Select non-local blocks (before and after current block)
        k_select = []
        v_select = []
        valid_select = []
        if block_idx > 0:
            k_select.append(k_all[:, :, :start_b, :])
            v_select.append(v_all[:, :, :start_b, :])
            valid_select.append(valid_mask[:, :start_b])
        if (block_idx + 1) * self.block_size < total_len:
            k_select.append(k_all[:, :, end_b:, :])
            v_select.append(v_all[:, :, end_b:, :])
            valid_select.append(valid_mask[:, end_b:])

        if not k_select:
            return torch.zeros_like(q_b)

        k_neighbors = torch.cat(k_select, dim=2)   # [B, H, L_neigh, D]
        v_neighbors = torch.cat(v_select, dim=2)
        valid_neighbors = torch.cat(valid_select, dim=1)   # [B, L_neigh]

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q_b, k_neighbors.transpose(-2, -1)) * scale   # [B, H, L_b, L_neigh]

        # Apply padding mask for neighbors
        pad_mask = ~valid_neighbors.unsqueeze(1).unsqueeze(2)   # [B,1,1,L_neigh]
        scores = scores.masked_fill(pad_mask, float("-inf"))

        top_k = min(self.top_k_inter, k_neighbors.size(2))
        _, topk_idx = torch.topk(scores, top_k, dim=-1)
        mask = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        probs = F.softmax(scores, dim=-1)
        return torch.matmul(probs, v_neighbors)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = hidden_states.shape
        H = self.num_heads

        # Convert mask to boolean valid mask [B, L]
        valid_mask = self._convert_mask_to_bool(attention_mask, B, L, hidden_states.device)

        q = self.W_q(hidden_states).view(B, L, H, self.head_dim).permute(0, 2, 1, 3)
        k = self.W_k(hidden_states).view(B, L, H, self.head_dim).permute(0, 2, 1, 3)
        v = self.W_v(hidden_states).view(B, L, H, self.head_dim).permute(0, 2, 1, 3)

        num_blocks = (L + self.block_size - 1) // self.block_size
        use_rectify = self.layer_idx % self.rectification_interval == 0
        self.layer_idx += 1

        if use_rectify:
            # Full attention with padding mask
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            pad_mask = ~valid_mask.unsqueeze(1).unsqueeze(2)   # [B,1,1,L]
            scores = scores.masked_fill(pad_mask, float("-inf"))
            attn_probs = F.softmax(scores, dim=-1)
            out = torch.matmul(attn_probs, v)
            if output_attentions:
                return_attn = attn_probs
            else:
                return_attn = None
        else:
            out_list = []
            for b_idx in range(num_blocks):
                start = b_idx * self.block_size
                end = min(start + self.block_size, L)
                q_b = q[:, :, start:end, :]
                k_b = k[:, :, start:end, :]
                v_b = v[:, :, start:end, :]
                block_valid = valid_mask[:, start:end]   # [B, L_b]
                block_pad_mask = ~block_valid

                intra_out = self._intra_block_attention(q_b, k_b, v_b, block_pad_mask, causal=True)
                inter_out = self._inter_block_sparse(q_b, k, v, b_idx, num_blocks, valid_mask)
                out_b = intra_out + inter_out
                out_list.append(out_b)
            out = torch.cat(out_list, dim=2)
            return_attn = None   # Sparse path does not produce full attention matrix

        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out), return_attn