"""
HashedNets: Weight sharing via hash function (efficient vectorized version).
Reference: "HashedNets: Efficient Training of Neural Networks with Hash Tables"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Large primes for hash mixing
_PRIME1 = 2654435761
_PRIME2 = 2246822519
_PRIME3 = 3266489917

def fast_hash(indices_i, indices_j, n_buckets):
    """Vectorized hash: (i * PRIME1 + j * PRIME2) % n_buckets."""
    return ((indices_i * _PRIME1 + indices_j * _PRIME2) % n_buckets).long()


class HashedLinear(nn.Module):
    """
    Linear layer with weight sharing via a hash table.

    The weight matrix W of shape (out_features, in_features) is generated
    on-the-fly from a small parameter table `weight_table` of size `n_buckets`.
    The mapping from (row, col) to bucket index uses a fast vectorized hash.
    """

    def __init__(self, in_features, out_features, n_buckets=1024, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_buckets = n_buckets

        # Weight table: shared values
        self.weight_table = nn.Parameter(torch.randn(n_buckets))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Precompute the hash indices for all (i, j) pairs
        # We store them as buffer so they don't change (in HashedNets the mapping is fixed)
        i_idx = torch.arange(out_features).unsqueeze(1).expand(-1, in_features)
        j_idx = torch.arange(in_features).unsqueeze(0).expand(out_features, -1)
        hash_idx = fast_hash(i_idx, j_idx, n_buckets)  # (out, in)
        self.register_buffer('hash_idx', hash_idx)

    def forward(self, x):
        # Reconstruct weight matrix from table and hash indices
        W = self.weight_table[self.hash_idx]  # (out, in)
        return F.linear(x, W, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, n_buckets: int) -> 'HashedLinear':
        """Create from an existing nn.Linear by hashing its weight matrix."""
        hashed = cls(linear.in_features, linear.out_features, n_buckets,
                     bias=linear.bias is not None)
        # Copy bias
        if linear.bias is not None:
            hashed.bias.data.copy_(linear.bias.data)
        # Initialize weight table by averaging the weights that fall into each bucket
        W = linear.weight.data.float()
        hash_idx = hashed.hash_idx
        for b in range(n_buckets):
            mask = (hash_idx == b)
            if mask.any():
                hashed.weight_table.data[b] = W[mask].mean()
        return hashed