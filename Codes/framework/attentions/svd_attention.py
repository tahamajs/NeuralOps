import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("svd")
class SVDLinearAttention(nn.Module):
    def __init__(self, config, rank=64, dropout=0.1):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.rank = rank
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # Use provided dropout or fallback to config.dropout
        dropout_val = (
            dropout if dropout is not None else getattr(config, "dropout", 0.1)
        )
        self.dropout = nn.Dropout(dropout_val)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        Q = self.transpose_for_scores(self.query(hidden_states))
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))

        # Fallback to full attention if rank not set or seq_len small
        if self.rank is None or seq_len <= self.rank:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
            if attention_mask is not None:
                # mask should be additive (0 for keep, -inf for pad)
                scores = scores + attention_mask
            probs = F.softmax(scores, dim=-1)
            probs = self.dropout(probs)
            context = torch.matmul(probs, V)
        else:
            # SVD compression (simplified – uses whole batch)
            # This part is kept as in original but with device handling
            hidden_cpu = hidden_states.cpu()
            U_r = []
            for b in range(batch_size):
                U, S, _ = torch.svd_lowrank(hidden_cpu[b], q=self.rank, niter=2)
                U_r.append(U[:, : self.rank])
            U_r = torch.stack(U_r, dim=0).to(hidden_states.device)
            U_r = U_r.unsqueeze(1)  # (B,1,seq_len,rank)
            K_proj = torch.matmul(U_r.transpose(-2, -1), K)  # (B,1,rank,D)
            V_proj = torch.matmul(U_r.transpose(-2, -1), V)
            scores = torch.matmul(Q, K_proj.transpose(-2, -1)) / (self.head_dim**0.5)
            if attention_mask is not None:
                scores = scores + attention_mask.unsqueeze(-1)
            probs = F.softmax(scores, dim=-1)
            probs = self.dropout(probs)
            context = torch.matmul(probs, V_proj)

        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, self.all_head_size)
        )
        return (context, None)  # always return tuple

    def get_ortho_loss(self):
        return torch.tensor(0.0, device=self.query.weight.device)
