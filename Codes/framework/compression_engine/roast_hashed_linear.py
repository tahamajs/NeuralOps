"""
ROAST: Hardware‑Aware Weight Sharing with Tiling.
Reference: "ROAST: A Hardware‑Aware Weight Sharing Scheme for Efficient Neural Networks"
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
        W_tiles = self.weight_table[self.tile_hash_idx]  # (tiles_out, tiles_in)
        W_full = torch.repeat_interleave(
            torch.repeat_interleave(W_tiles, self.tile_size, dim=0),
            self.tile_size,
            dim=1,
        )
        return W_full[: self.out_features, : self.in_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized forward pass without per-element python loops."""
        W = self.weight.to(device=x.device, dtype=x.dtype)
        return F.linear(x, W, self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        n_buckets: int = 1024,
        tile_size: int = 32,
    ) -> "ROASTLinear":
        """Initialize from a pretrained nn.Linear or Conv1D."""
        raw_w = linear.weight.data
        cls_name = linear.__class__.__name__
        is_conv1d = "conv1d" in cls_name.lower() or (
            raw_w.ndim == 2
            and hasattr(linear, "in_features")
            and linear.in_features == raw_w.shape[0]
            and raw_w.shape[0] < raw_w.shape[1]
        )

        if is_conv1d:
            W = raw_w.t().float()
            out_f, in_f = W.shape
        else:
            W = raw_w.float()
            out_f, in_f = W.shape

        bias = getattr(linear, "bias", None) is not None
        roast = cls(
            in_features=in_f,
            out_features=out_f,
            n_buckets=n_buckets,
            tile_size=tile_size,
            bias=bias,
        )
        if bias and linear.bias is not None:
            roast.bias.data.copy_(linear.bias.data)

        ts = tile_size
        tiles_out = (out_f + ts - 1) // ts
        tiles_in = (in_f + ts - 1) // ts

        # Gather per-tile means
        for i in range(tiles_out):
            for j in range(tiles_in):
                tile = W[
                    i * ts : min((i + 1) * ts, out_f),
                    j * ts : min((j + 1) * ts, in_f),
                ]
                bucket_idx = roast.tile_hash_idx[i, j].item()
                roast.weight_table.data[bucket_idx] = tile.mean()

        return roast


from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear


@COMPRESSOR_REGISTRY.register("roast")
@COMPRESSOR_REGISTRY.register("roast_compressor")
class ROASTCompressor(BaseCompressor):
    """
    Replace eligible linear and Conv1D layers with ROASTLinear (hardware‑aware weight sharing).
    """

    def __init__(
        self,
        model: nn.Module,
        n_buckets: int = 1024,
        tile_size: int = 32,
        target_modules: Optional[list[str]] = None,
        max_error: float = 0.50,
        fallback_to_original: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="roast", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            n_buckets = getattr(cfg, "n_buckets", n_buckets)
            tile_size = getattr(cfg, "tile_size", tile_size)
            max_error = getattr(cfg, "max_error", max_error)
            fallback_to_original = getattr(cfg, "fallback_to_original", fallback_to_original)

        self.n_buckets = int(n_buckets)
        self.tile_size = int(tile_size)
        self.target_modules = target_modules
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.compressed_layers: list[dict[str, Any]] = []

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if any(k in name.lower() for k in ["classifier", "score", "lm_head", "pooler", "embedding"]):
            return False
        cls_name = module.__class__.__name__
        is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name.lower()
        if not is_linear_like:
            return False
        if isinstance(module, ROASTLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress the target module using ROAST tiling."""
        self.compressed_layers = []
        for name, module in list(self.model.named_modules()):
            if self._is_target_module(name, module):
                raw_w = module.weight.data
                cls_name = module.__class__.__name__
                is_conv1d = "conv1d" in cls_name.lower() or (
                    raw_w.ndim == 2
                    and hasattr(module, "in_features")
                    and module.in_features == raw_w.shape[0]
                    and raw_w.shape[0] < raw_w.shape[1]
                )
                W_orig = raw_w.t().float() if is_conv1d else raw_w.float()

                new_layer = ROASTLinear.from_linear(
                    module, self.n_buckets, self.tile_size
                )
                W_rec = new_layer.weight.float()
                err = (torch.norm(W_orig - W_rec, p="fro") / (torch.norm(W_orig, p="fro") + 1e-8)).item()

                if self.fallback_to_original and err > self.max_error:
                    continue

                replace_module(self.model, name, new_layer)
                self.compressed_layers.append({
                    "name": name,
                    "relative_error": float(err),
                })

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "compressed_layer_count": len(self.compressed_layers),
            "n_buckets": self.n_buckets,
            "tile_size": self.tile_size,
            "compressed_layers": self.compressed_layers,
        })
        return self.model, stats


@COMPRESSOR_REGISTRY.register("roast_factory")
def roast_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """ROAST compressor factory function."""
    compressor = ROASTCompressor(model, comp_config=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model

