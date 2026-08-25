"""
ROAST: Hardware‑Aware Weight Sharing with Tiling.
Reference: "ROAST: A Hardware‑Aware Weight Sharing Scheme for Efficient Neural Networks"
"""
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor


class ROASTLinear(nn.Module):
    """
    Linear layer with weight sharing based on tiles (blocks) rather than
    individual scalar weights. Each tile of size (tile_size, tile_size) shares
    a single weight from a small table.
    """

    def __init__(
        self, in_features, out_features, n_buckets=1024, tile_size=32, bias=True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tile_size = tile_size
        self.n_buckets = n_buckets

        # Number of tiles along each dimension (padded)
        self.tiles_out = (out_features + tile_size - 1) // tile_size
        self.tiles_in = (in_features + tile_size - 1) // tile_size
        self.tiles_out * self.tiles_in

        # Weight table: one scalar per bucket, but each tile repeats that scalar
        # To have unique per-tile scaling, we can also store a full tile pattern
        # but the simplest ROAST uses one weight per tile.
        self.weight_table = nn.Parameter(torch.randn(n_buckets))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Precompute tile indices
        tile_idx_i = torch.arange(self.tiles_out).unsqueeze(1).expand(-1, self.tiles_in)
        tile_idx_j = torch.arange(self.tiles_in).unsqueeze(0).expand(self.tiles_out, -1)
        # Use a simple hash to map each tile to a bucket
        self.register_buffer(
            "tile_hash_idx",
            ((tile_idx_i * 10007 + tile_idx_j * 10009) % n_buckets).long(),
        )

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        out_padded = self.tiles_out * self.tile_size
        in_padded = self.tiles_in * self.tile_size
        W = torch.zeros(out_padded, in_padded, device=self.weight_table.device, dtype=self.weight_table.dtype)
        for i in range(self.tiles_out):
            for j in range(self.tiles_in):
                val = self.weight_table[self.tile_hash_idx[i, j]]
                W[
                    i * self.tile_size : (i + 1) * self.tile_size,
                    j * self.tile_size : (j + 1) * self.tile_size,
                ] = val
        return W[: self.out_features, : self.in_features]

    def forward(self, x):
        # Build weight matrix from tiles
        # Start with a zero matrix of padded size
        """Forward pass."""
        out_padded = self.tiles_out * self.tile_size
        in_padded = self.tiles_in * self.tile_size
        W = torch.zeros(out_padded, in_padded, device=x.device, dtype=x.dtype)

        # Fill each tile with its bucket value
        for i in range(self.tiles_out):
            for j in range(self.tiles_in):
                val = self.weight_table[self.tile_hash_idx[i, j]]
                W[
                    i * self.tile_size : (i + 1) * self.tile_size,
                    j * self.tile_size : (j + 1) * self.tile_size,
                ] = val

        # Crop to actual dimensions
        W = W[: self.out_features, : self.in_features]
        return F.linear(x, W, self.bias)

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, n_buckets: int = 1024, tile_size: int = 32
    ) -> "ROASTLinear":
        """Initialize from a pretrained nn.Linear."""
        roast = cls(
            linear.in_features,
            linear.out_features,
            n_buckets,
            tile_size,
            bias=linear.bias is not None,
        )
        if linear.bias is not None:
            roast.bias.data.copy_(linear.bias.data)
        # Compute mean weight per tile and assign to bucket
        W = linear.weight.data.float()
        out_f, in_f = W.shape
        ts = tile_size
        tiles_out = (out_f + ts - 1) // ts
        tiles_in = (in_f + ts - 1) // ts
        # Gather per-tile means
        for i in range(tiles_out):
            for j in range(tiles_in):
                tile = W[
                    i * ts : min((i + 1) * ts, out_f), j * ts : min((j + 1) * ts, in_f)
                ]
                bucket_idx = roast.tile_hash_idx[i, j].item()
                roast.weight_table.data[bucket_idx] = tile.mean()
        return roast


class ROASTCompressor(BaseCompressor):
    """
    Replace all nn.Linear layers with ROASTLinear (hardware‑aware weight sharing).
    """

    def __init__(self, model, n_buckets=1024, tile_size=32, **kwargs):
        super().__init__(model, name="roast", **kwargs)
        self.n_buckets = n_buckets
        self.tile_size = tile_size

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                new_layer = ROASTLinear.from_linear(
                    module, self.n_buckets, self.tile_size
                )
                parent = self._get_parent(name)
                if parent is not None:
                    setattr(parent, name.split(".")[-1], new_layer)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()

    def _get_parent(self, full_name) -> Any:
        """ get parent."""
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None
        return parent
