"""
CURLinear: drop‑in replacement for nn.Linear using CUR decomposition.
"""

import torch
import torch.nn as nn
import numpy as np


class CURLinear(nn.Module):
    """
    Linear layer whose weight is approximated as W ≈ C @ U @ R,
    using leverage score sampling for column/row selection.
    """

    def __init__(self, linear: nn.Linear, rank: int, method: str = 'leverage'):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank

        W = linear.weight.data.float().cpu().numpy()
        out_dim, in_dim = W.shape
        r = min(rank, min(out_dim, in_dim))

        # Column selection
        if method == 'leverage':
            col_norms = np.linalg.norm(W, axis=0)
            col_probs = col_norms / (col_norms.sum() + 1e-8)
        else:
            col_probs = np.ones(in_dim) / in_dim
        selected_cols = np.random.choice(in_dim, size=r, replace=False, p=col_probs)
        C = torch.tensor(W[:, selected_cols], dtype=torch.float32)

        # Row selection
        if method == 'leverage':
            row_norms = np.linalg.norm(W, axis=1)
            row_probs = row_norms / (row_norms.sum() + 1e-8)
        else:
            row_probs = np.ones(out_dim) / out_dim
        selected_rows = np.random.choice(out_dim, size=r, replace=False, p=row_probs)
        R = torch.tensor(W[selected_rows, :], dtype=torch.float32)

        # Compute U
        C_pinv = torch.linalg.pinv(C)
        R_pinv = torch.linalg.pinv(R)
        U = C_pinv @ torch.tensor(W, dtype=torch.float32) @ R_pinv

        self.register_buffer('C', C)
        self.register_buffer('U', U)
        self.register_buffer('R', R)

        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        W_approx = self.C @ self.U @ self.R
        y = x @ W_approx.T
        if self.bias is not None:
            y = y + self.bias
        return y