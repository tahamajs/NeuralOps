# framework/attentions/specialized/block_transformer_hiba.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.attentions.core.base import BaseAttention
from framework.attentions.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("block_transformer_hiba")
class BlockTransformerHIBA(BaseAttention):
    """
    Block Transformer with HIBA: intra‑block full attention + inter‑block sparse attention
    and periodic dense rectification to prevent fragmentation.
    """
    def __init__(self, config, rank=64, block_size=512, top_k_inter=32, rectification_interval=4):
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

        self.layer_idx = 0   # incremented each forward pass

    def _intra_block_attention(self, q, k, v, causal=True):
        """Standard scaled dot‑product attention, optionally causal."""
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if causal:
            L = q.size(2)
            mask = torch.tril(torch.ones(L, L, device=q.device)).bool()
            attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        return torch.matmul(F.softmax(attn, dim=-1), v)

    def _inter_block_sparse(self, q_b, k_all, v_all, block_idx, num_blocks):
        """Sparse attention to top‑k tokens from adjacent blocks."""
        B, H, _, D = q_b.shape
        total_len = k_all.size(2)
        start_b = block_idx * self.block_size
        end_b = min((block_idx+1)*self.block_size, total_len)

        k_select = []
        v_select = []
        if block_idx > 0:
            k_select.append(k_all[:, :, :start_b, :])
            v_select.append(v_all[:, :, :start_b, :])
        if (block_idx+1)*self.block_size < total_len:
            k_select.append(k_all[:, :, end_b:, :])
            v_select.append(v_all[:, :, end_b:, :])

        if not k_select:
            return torch.zeros_like(q_b)
        k_neighbors = torch.cat(k_select, dim=2)
        v_neighbors = torch.cat(v_select, dim=2)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q_b, k_neighbors.transpose(-2, -1)) * scale
        top_k = min(self.top_k_inter, k_neighbors.size(2))
        _, topk_idx = torch.topk(scores, top_k, dim=-1)
        mask = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        return torch.matmul(F.softmax(scores, dim=-1), v_neighbors)

    def forward(self, x, mask=None, **kwargs):
        B, L, D = x.shape
        H = self.num_heads
        head_dim = self.head_dim

        q = self.W_q(x).view(B, L, H, head_dim).permute(0,2,1,3)  # (B, H, L, d)
        k = self.W_k(x).view(B, L, H, head_dim).permute(0,2,1,3)
        v = self.W_v(x).view(B, L, H, head_dim).permute(0,2,1,3)

        num_blocks = (L + self.block_size - 1) // self.block_size
        use_rectify = (self.layer_idx % self.rectification_interval == 0)
        self.layer_idx += 1

        if use_rectify:
            # Global full attention (dense rectification)
            scale = head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask.unsqueeze(1)
            attn_probs = F.softmax(scores, dim=-1)
            out = torch.matmul(attn_probs, v)
        else:
            out_list = []
            for b_idx in range(num_blocks):
                start = b_idx * self.block_size
                end = min(start + self.block_size, L)
                q_b = q[:, :, start:end, :]
                k_b = k[:, :, start:end, :]
                v_b = v[:, :, start:end, :]

                intra_out = self._intra_block_attention(q_b, k_b, v_b, causal=True)
                inter_out = self._inter_block_sparse(q_b, k, v, b_idx, num_blocks)
                out_b = intra_out + inter_out
                out_list.append(out_b)
            out = torch.cat(out_list, dim=2)   # (B, H, L, d)

        out = out.permute(0,2,1,3).contiguous().view(B, L, D)
        return self.out_proj(out), None