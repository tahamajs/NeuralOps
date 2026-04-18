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
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Apply SVD to weights once at initialization
        if rank is not None and rank < config.hidden_size:
            self._apply_svd_to_weights()

    def _apply_svd_to_weights(self):
        """Reduce rank of query, key, value weight matrices using SVD"""
        with torch.no_grad():
            for linear in [self.query, self.key, self.value]:
                U, S, V = torch.svd(linear.weight.data)
                U_r = U[:, :self.rank]
                S_r = S[:self.rank]
                V_r = V[:, :self.rank]
                linear.weight.data = U_r @ torch.diag(S_r) @ V_r.T

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False):

        if encoder_hidden_states is not None:
            raise NotImplementedError("SVDLinearAttention does not support cross-attention yet.")

        Q = self.transpose_for_scores(self.query(hidden_states))
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context = torch.matmul(attention_probs, V)

        context = context.permute(0, 2, 1, 3).contiguous()
        new_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_shape)

        if output_attentions:
            return (context, attention_probs)
        else:
            return (context,)