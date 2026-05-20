import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("nsa")
class NativeSparseAttention(nn.Module):
    def __init__(self, config, window_size=128, top_k=32):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.window_size = window_size
        self.top_k = top_k
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0,2,1,3)
        K = self.k_proj(hidden_states).view(B, L, H, d).permute(0,2,1,3)
        V = self.v_proj(hidden_states).view(B, L, H, d).permute(0,2,1,3)

        scale = d ** -0.5
        idx = torch.arange(L, device=Q.device)
        local_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size
        local_mask = local_mask.unsqueeze(0).unsqueeze(0)   # (1,1,L,L)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        topk_mask = torch.zeros_like(scores)

        for i in range(L):
            s_i = scores[:, :, i, :]
            s_i = s_i.masked_fill(local_mask[0,0,i].unsqueeze(0).unsqueeze(0), float('-inf'))
            _, topk_ind = torch.topk(s_i, min(self.top_k, L), dim=-1)
            topk_mask[:, :, i, :].scatter_(-1, topk_ind, 1.0)

        topk_mask = topk_mask + local_mask.float()
        scores = scores.masked_fill(topk_mask == 0, float('-inf'))

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
            else:
                mask = attention_mask
            scores = scores + (1.0 - mask) * -1e9

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V).permute(0,2,1,3).contiguous().view(B, L, D)
        return self.out_proj(context), None
    