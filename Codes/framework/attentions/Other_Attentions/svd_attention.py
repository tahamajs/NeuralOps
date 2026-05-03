import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base_attention import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("svd")
class SVDLinearAttention(BaseAttention):
    def __init__(self, config, rank=64):
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        Q = self.transpose_for_scores(self.query(hidden_states))
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))

        # If no compression or sequence too short, use full attention
        if self.rank is None or seq_len <= self.rank:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if attention_mask is not None:
                scores = scores + attention_mask
            probs = F.softmax(scores, dim=-1)
            probs = self.dropout(probs)
            context = torch.matmul(probs, V)
        else:
            # SVD compression – move to CPU if needed
            original_device = hidden_states.device
            if original_device.type == 'mps':
                hidden_cpu = hidden_states.cpu()
            else:
                hidden_cpu = hidden_states

            U_r_list = []
            for b in range(batch_size):
                U, S, _ = torch.svd_lowrank(hidden_cpu[b], q=self.rank, niter=2)
                U_r_list.append(U[:, :self.rank])
            U_r = torch.stack(U_r_list, dim=0).to(original_device)
            U_r = U_r.unsqueeze(1)   # (batch, 1, seq_len, rank)

            K_proj = torch.matmul(U_r.transpose(-2, -1), K)
            V_proj = torch.matmul(U_r.transpose(-2, -1), V)

            scores = torch.matmul(Q, K_proj.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if attention_mask is not None:
                scores = scores + attention_mask.unsqueeze(-1)
            probs = F.softmax(scores, dim=-1)
            probs = self.dropout(probs)
            context = torch.matmul(probs, V_proj)

        context = context.permute(0, 2, 1, 3).contiguous()
        new_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_shape)
        return (context, None) if kwargs.get('output_attentions', False) else (context,)