import torch
import torch.nn as nn
import numpy as np

class SVDLinearLayer(nn.Module):
    """Replace a linear layer with low‑rank approximation U * S * V."""
    def __init__(self, original_layer: nn.Linear, rank: int):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = min(rank, self.in_features, self.out_features)

        # Perform SVD on the weight matrix
        W = original_layer.weight.data.cpu().numpy()
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        # Keep top rank components
        self.U = nn.Parameter(torch.from_numpy(U[:, :self.rank]).float())
        self.S = nn.Parameter(torch.from_numpy(S[:self.rank]).float())
        self.Vt = nn.Parameter(torch.from_numpy(Vt[:self.rank, :]).float())

        # Bias
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        # Reconstruct weight: U * diag(S) * Vt
        # Using decomposition to reduce FLOPs: (x * Vt^T) * (U * diag(S))
        x_v = torch.matmul(x, self.Vt.t())
        sv = self.U.t() * self.S.unsqueeze(0)
        out = torch.matmul(x_v, sv.t())
        if self.bias is not None:
            out += self.bias
        return out