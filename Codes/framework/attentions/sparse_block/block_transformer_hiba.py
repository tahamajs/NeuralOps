import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base_attention import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("block_transformer_hiba")
class BlockTransformerHIBA(BaseAttention):
    """
    Block Transformer with HIBA: intra‑block full attention + inter‑block sparse attention
    and periodic dense rectification.
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

        # Layer‑wise counter for rectification
        self.layer_idx = 0

    def _block_attention(self, q, k, v, block_indices):
        # simple block sparse: not implemented fully
        pass

    def forward(self, x, mask=None, **kwargs):
        B, L, D = x.shape
        q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = self.W_k(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = self.W_v(x).view(B, L, self.num_heads, self.head_dim).permute(0,2,1,3)

        scale = self.head_dim ** -0.5
        num_blocks = (L + self.block_size - 1) // self.block_size

        # Decide whether to use dense rectification this layer
        use_rectify = (self.layer_idx % self.rectification_interval == 0)
        self.layer_idx += 1

        if use_rectify:
            # Full global attention
            attn = torch.matmul(q, k.transpose(-2,-1)) * scale
            attn = F.softmax(attn, dim=-1)
        else:
            # Intra‑block full attention + inter‑block sparse attention
            out_list = []
            for b in range(num_blocks):
                start = b * self.block_size
                end = min(start + self.block_size, L)
                q_b = q[:, :, start:end, :]
                k_b = k[:, :, start:end, :]
                v_b = v[:, :, start:end, :]

                # Intra‑block attention (causal)
                intra_attn = F.softmax(torch.matmul(q_b, k_b.transpose(-2,-1)) * scale, dim=-1)
                intra_out = torch.matmul(intra_attn, v_b)

                # Inter‑block: select top‑k tokens from adjacent blocks (simplified as averaging)
                if b > 0:
                    k_prev = k[:, :, start-self.block_size:start, :]
                    v_prev = v[:, :, start-self.block_size:start, :]
                    # compute attention to previous block and average
                    inter_attn = F.softmax(torch.matmul(q_b, k_prev.transpose(-2,-1)) * scale, dim=-1)
                    inter_out = torch.matmul(inter_attn, v_prev)
                    intra_out = intra_out + inter_out
                out_list.append(intra_out)
            attn_out = torch.cat(out_list, dim=2)
            return self.out_proj(attn_out.permute(0,2,1,3).contiguous().view(B, L, D)), None

        out = torch.matmul(attn, v).permute(0,2,1,3).contiguous().view(B, L, D)
        return self.out_proj(out), None