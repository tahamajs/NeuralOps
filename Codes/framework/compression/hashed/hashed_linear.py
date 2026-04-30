# framework/compression/hashed_linear.py
import torch
import torch.nn as nn
import hashlib

class HashedLinear(nn.Module):
    """
    HashedNets: Weight sharing via hash function.
    """
    def __init__(self, in_features, out_features, n_buckets=1024):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_buckets = n_buckets
        self.weight_table = nn.Parameter(torch.randn(n_buckets))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def _hash(self, i, j):
        s = f"{i}_{j}".encode()
        return int(hashlib.md5(s).hexdigest(), 16) % self.n_buckets

    def forward(self, x):
        W = torch.zeros(self.out_features, self.in_features, device=x.device)
        for i in range(self.out_features):
            for j in range(self.in_features):
                W[i, j] = self.weight_table[self._hash(i, j)]
        return x @ W.T + self.bias