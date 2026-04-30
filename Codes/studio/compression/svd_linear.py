# studio/compression/svd_linear.py
import torch
import torch.nn as nn
import numpy as np

class SVDLinearLayer(nn.Module):
    def __init__(self, linear_layer: nn.Linear, rank: int):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = min(rank, self.in_features, self.out_features)
        W = linear_layer.weight.data.cpu().numpy()
        U, S, Vh = np.linalg.svd(W, full_matrices=False)
        self.U = nn.Parameter(torch.from_numpy(U[:, :self.rank]).float())
        self.S = nn.Parameter(torch.from_numpy(S[:self.rank]).float())
        self.Vh = nn.Parameter(torch.from_numpy(Vh[:self.rank, :]).float())
        self.bias = nn.Parameter(linear_layer.bias.data.clone()) if linear_layer.bias is not None else None

    def forward(self, x):
        x_v = torch.matmul(x, self.Vh.t())
        sv = self.U.t() * self.S.unsqueeze(0)
        out = torch.matmul(x_v, sv.t())
        if self.bias is not None:
            out += self.bias
        return out