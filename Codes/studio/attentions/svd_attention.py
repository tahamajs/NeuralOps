# studio/attentions/svd_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SVDLinearAttention(nn.Module):
    def __init__(self, config, rank=64):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.rank = rank
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _reshape(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len = hidden_states.shape[:2]
        Q = self._reshape(self.query(hidden_states), batch_size)
        K = self._reshape(self.key(hidden_states), batch_size)
        V = self._reshape(self.value(hidden_states), batch_size)

        # Calculate low‑rank projection on the fly for demonstration
        with torch.no_grad():
            U, S, _ = torch.linalg.svd(hidden_states.view(batch_size, seq_len, -1).mean(dim=1), full_matrices=False)
            proj = U[:, :self.rank]  # (batch, hidden, rank)
            K_proj = torch.einsum('bhd,bnhd->bhnr', proj, K)  # simplified

        scores = torch.matmul(Q, K_proj.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1)
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        context = torch.matmul(probs, V)
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.all_head_size)
        return (self.out_proj(context), None)