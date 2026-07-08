"""SVD-based low-rank linear layer replacement."""

import numpy as np
import torch
import torch.nn as nn


class SVDLinearLayer(nn.Module):
    """Replace a linear layer with U @ diag(S) @ V, keeping top-`rank` components."""

    def __init__(self, linear_layer: nn.Linear, rank: int):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = min(rank, self.in_features, self.out_features)

        W = linear_layer.weight.data.float().cpu().numpy()
        U, S, Vh = np.linalg.svd(W, full_matrices=False)
        self.U = nn.Parameter(torch.from_numpy(U[:, : self.rank]).float())
        self.S = nn.Parameter(torch.from_numpy(S[: self.rank]).float())
        self.Vh = nn.Parameter(torch.from_numpy(Vh[: self.rank, :]).float())

        if linear_layer.bias is not None:
            self.bias = nn.Parameter(linear_layer.bias.data.clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        # (x @ Vh^T) * diag(S) @ U^T
        x_v = torch.matmul(x, self.Vh.t())
        out = torch.matmul(x_v * self.S.unsqueeze(0), self.U.t())
        if self.bias is not None:
            out = out + self.bias
        return out
