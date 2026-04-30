# framework/compression/roast_hashed_linear.py
import torch
import torch.nn as nn

class ROASTLinear(nn.Module):
    """
    ROAST: Hardware-aware hashing with tile-based sharing.
    """
    def __init__(self, in_features, out_features, n_buckets=1024, tile_size=32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tile_size = tile_size
        self.n_tiles = (in_features * out_features + tile_size - 1) // tile_size
        self.weight_table = nn.Parameter(torch.randn(self.n_tiles))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def _tile_index(self, i, j):
        return ((i // self.tile_size) * (self.out_features // self.tile_size) + (j // self.tile_size)) % self.n_tiles

    def forward(self, x):
        W = torch.zeros(self.out_features, self.in_features, device=x.device)
        for i in range(self.out_features):
            for j in range(self.in_features):
                W[i, j] = self.weight_table[self._tile_index(i, j)]
        return x @ W.T + self.bias