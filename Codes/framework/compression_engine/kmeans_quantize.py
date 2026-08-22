"""KMeans-based weight clustering quantizer with STE. Registered as 'kmeans'."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class KMeansSTEFunction(torch.autograd.Function):
    """
    Straight-Through Estimator for codebook lookup.
    Forward projects centroid values by index; backward passes gradients directly to centroids.
    """

    @staticmethod
    def forward(
        ctx: Any,
        centroids: torch.Tensor,
        indices: torch.Tensor,
        x: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ctx.save_for_backward(centroids, indices, x, bias)
        W = centroids[indices].to(dtype=x.dtype)
        return F.linear(x, W, bias)

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], None, Optional[torch.Tensor], Optional[torch.Tensor]]:
        centroids, indices, x, bias = ctx.saved_tensors
        W = centroids[indices].to(dtype=x.dtype)

        # Gradient w.r.t input x
        grad_x = grad_output.matmul(W) if ctx.needs_input_grad[2] else None

        # Gradient w.r.t bias
        grad_bias = None
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(dim=list(range(grad_output.ndim - 1)))

        # Gradient w.r.t centroids
        grad_centroids = None
        if ctx.needs_input_grad[0]:
            if x.ndim == 2:
                grad_W = grad_output.t().matmul(x)
            else:
                grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])
                x_flat = x.reshape(-1, x.shape[-1])
                grad_W = grad_output_flat.t().matmul(x_flat)

            grad_centroids = torch.zeros_like(centroids)
            flat_indices = indices.view(-1)
            flat_grad_W = grad_W.view(-1).to(dtype=centroids.dtype, device=centroids.device)
            grad_centroids.scatter_add_(0, flat_indices, flat_grad_W)

        return grad_centroids, None, grad_x, grad_bias


def fast_kmeans_1d(
    tensor: torch.Tensor,
    n_clusters: int = 256,
    max_iters: int = 20,
    tol: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast 1D Lloyd's algorithm on PyTorch tensor with linear quantile initialization.
    Returns (centroids, indices).
    """
    flat = tensor.detach().float().view(-1)
    numel = flat.numel()
    k = min(n_clusters, numel)

    if k <= 1:
        c = flat.mean().unsqueeze(0)
        idx = torch.zeros(tensor.shape, dtype=torch.long, device=tensor.device)
        return c, idx

    # Quantile-based centroid initialization
    sorted_vals, _ = torch.sort(flat)
    quantiles = torch.linspace(0, numel - 1, steps=k, device=flat.device).long()
    centroids = sorted_vals[quantiles].clone()

    for _ in range(max_iters):
        # Compute distances: |x_i - c_j|
        # For 1D, use searchsorted with midpoints between sorted centroids
        sorted_c, sort_perm = torch.sort(centroids)
        midpoints = 0.5 * (sorted_c[:-1] + sorted_c[1:])
        sorted_idx = torch.searchsorted(midpoints, flat)
        idx = sort_perm[sorted_idx]

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=flat.device)
        counts.scatter_add_(0, idx, torch.ones_like(flat))
        new_centroids.scatter_add_(0, idx, flat)

        non_empty = counts > 0
        new_centroids[non_empty] = new_centroids[non_empty] / counts[non_empty]
        new_centroids[~non_empty] = centroids[~non_empty]

        shift = torch.max(torch.abs(new_centroids - centroids)).item()
        centroids = new_centroids
        if shift < tol:
            break

    # Final assignment
    sorted_c, sort_perm = torch.sort(centroids)
    midpoints = 0.5 * (sorted_c[:-1] + sorted_c[1:])
    sorted_idx = torch.searchsorted(midpoints, flat)
    indices = sort_perm[sorted_idx].reshape(tensor.shape)

    return centroids, indices


class KMeansLinear(nn.Module):
    """Linear layer with weights represented by k-means centroids and discrete index table."""

    def __init__(self, linear: nn.Linear, n_clusters: int = 256):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.n_clusters = n_clusters

        centroids, indices = fast_kmeans_1d(linear.weight.data, n_clusters=n_clusters)
        self.centroids = nn.Parameter(centroids.to(dtype=linear.weight.dtype))
        self.register_buffer("indices", indices)

        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return KMeansSTEFunction.apply(
                self.centroids,
                self.indices,
                x,
                self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None,
            )
        W = self.centroids[self.indices].to(dtype=x.dtype, device=x.device)
        return F.linear(
            x,
            W,
            self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None,
        )

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, n_clusters={self.n_clusters}"


class KMeansCompressor(BaseCompressor):
    """
    KMeansCompressor: Vector quantization via scalar k-means clustering of weight parameters.

    Args:
        model: PyTorch model.
        n_clusters: Number of centroid codebook entries (default 256 -> 8 bits, 16 -> 4 bits).
        bits: Optional bitwidth constraint (e.g. 4 or 8) which sets n_clusters = 2**bits.
    """

    def __init__(
        self,
        model: nn.Module,
        n_clusters: int = 256,
        bits: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="kmeans", **kwargs)
        if bits is not None:
            self.n_clusters = 2 ** int(bits)
        else:
            self.n_clusters = int(n_clusters)
        self.bits = math.ceil(math.log2(max(self.n_clusters, 2)))

    def _replace_module(self, full_name: str, new_module: nn.Module) -> None:
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Replace all nn.Linear modules with KMeansLinear layers."""
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear):
                new_layer = KMeansLinear(module, self.n_clusters)
                self._replace_module(name, new_layer)

        stats = self.get_compression_stats()
        stats["n_clusters"] = self.n_clusters
        stats["effective_bits_per_weight"] = self.bits
        stats["theoretical_vram_saving"] = f"{(1.0 - self.bits / 16.0) * 100:.1f}% vs FP16"
        return self.model, stats


@COMPRESSOR_REGISTRY.register("kmeans")
def kmeans_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """KMeans compressor factory function."""
    n_clusters = kwargs.pop("n_clusters", getattr(comp_config, "kmeans_clusters", 256) if comp_config is not None else 256)
    bits = kwargs.pop("bits", getattr(comp_config, "kmeans_bits", None) if comp_config is not None else None)
    compressor = KMeansCompressor(model, n_clusters=n_clusters, bits=bits, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
