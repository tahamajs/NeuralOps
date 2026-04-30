# framework/compression/cur_linear.py
import torch
import torch.nn as nn
import numpy as np

class CURLinear(nn.Module):
    """
    CUR decomposition: W ≈ C * U * R.
    """
    def __init__(self, linear: nn.Linear, rank: int):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        W = linear.weight.data.cpu().numpy()
        # Use leverage scores to select columns and rows
        # Simplified: random sampling
        col_idx = np.random.choice(W.shape[1], rank, replace=False)
        row_idx = np.random.choice(W.shape[0], rank, replace=False)
        C = torch.tensor(W[:, col_idx], dtype=torch.float32)
        R = torch.tensor(W[row_idx, :], dtype=torch.float32)
        # Solve for U: C * U ≈ W * inv(R) ? Actually we want C * U * R ≈ W.
        # Solve for U in C * U ≈ W * pinv(R)
        R_pinv = torch.linalg.pinv(R)
        U = torch.linalg.lstsq(C, W @ R_pinv).solution
        self.register_buffer('C', C)
        self.register_buffer('U', U)
        self.register_buffer('R', R)
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None

    def forward(self, x):
        W_approx = self.C @ self.U @ self.R
        y = x @ W_approx.T
        if self.bias is not None:
            y += self.bias
        return y