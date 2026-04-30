# framework/compression/rpc_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base_attention import BaseAttention
from ..core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("rpc")
class RPCAttention(BaseAttention):
    """
    RPC Attention: Kernel PCA perspective on self-attention.
    """
    def __init__(self, config, rank=64):
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.rank = rank

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)

    def _reshape(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        Q = self._reshape(self.query(hidden_states), batch_size)
        K = self._reshape(self.key(hidden_states), batch_size)
        V = self._reshape(self.value(hidden_states), batch_size)

        # Kernel PCA: compute principal components of K in feature space
        # Using RBF kernel approximation
        K_flat = K.reshape(batch_size, self.num_heads, seq_len, -1)
        # Compute Gram matrix (N x N) per head, approximated by Nyström
        # Simplified: use linear kernel (standard PCA)
        U, S, _ = torch.linalg.svd(K_flat, full_matrices=False)
        U_r = U[:, :, :, :self.rank]
        # Project queries onto the principal components
        scores = torch.einsum('bqhd,bhrd->bqhr', Q, U_r) / (self.head_dim ** 0.5)
        probs = F.softmax(scores, dim=-1)
        context = torch.einsum('bqhr,bhrd->bqhd', probs, V)
        context = context.permute(0,2,1,3).contiguous().view(batch_size, seq_len, self.all_head_size)
        return self.out_proj(context), None