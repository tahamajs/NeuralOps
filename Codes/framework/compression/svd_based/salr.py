# framework/compression/salr.py
import torch
import torch.nn as nn

class SALRLayer(nn.Module):
    """
    SALR: Unifies low-rank and sparse pruning.
    """
    def __init__(self, linear: nn.Linear, rank: int, sparsity: float):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.sparsity = sparsity
        U, S, Vt = torch.linalg.svd(linear.weight.data, full_matrices=False)
        self.U = nn.Parameter(U[:, :rank])
        self.S = nn.Parameter(S[:rank])
        self.Vt = nn.Parameter(Vt[:rank, :])
        self.bias = nn.Parameter(linear.bias.data.clone() if linear.bias is not None else torch.zeros(out_features))
        # Sparse mask for low-rank factors
        self.register_buffer('mask', (torch.rand(rank, rank) < sparsity).float())

    def forward(self, x):
        S_sparse = self.S * self.mask[:self.rank, :self.rank]
        W = self.U @ torch.diag(S_sparse) @ self.Vt
        return x @ W.T + self.bias