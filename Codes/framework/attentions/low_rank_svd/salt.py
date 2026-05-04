import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.registry import AUX_REGISTRY

@AUX_REGISTRY.register("salt_linear")
class SALTLinear(nn.Module):
    """Singular Value Adaptation for a linear layer."""
    def __init__(self, in_features, out_features, rank=16):
        super().__init__()
        # Original weight frozen
        self.register_buffer('original_weight', torch.randn(out_features, in_features))
        # Decompose original weight via SVD (frozen)
        U, S, Vt = torch.linalg.svd(self.original_weight, full_matrices=False)
        self.register_buffer('U', U)
        self.register_buffer('Vt', Vt)
        self.S = nn.Parameter(S)                # trainable copy of singular values
        self.gamma = nn.Parameter(torch.ones(rank))
        # Low‑rank update matrices
        self.A = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, in_features) * 0.01)

    def forward(self, x):
        # Adapted weight = U @ diag(S + gamma * S[:rank]) @ Vt + A @ B
        adapted_S = self.S[:self.gamma.shape[0]] * self.gamma  # scale top-k
        adapted_weight = self.U[:, :self.gamma.shape[0]] @ torch.diag(adapted_S) @ self.Vt[:self.gamma.shape[0], :]
        low_rank = self.A @ self.B
        W = adapted_weight + low_rank
        return F.linear(x, W)
