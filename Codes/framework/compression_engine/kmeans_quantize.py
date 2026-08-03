"""KMeans‑based weight clustering compressor. Registered as 'kmeans'."""

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from sklearn.cluster import KMeans


class KMeansLinear(nn.Module):
    """Linear layer with weights replaced by k‑means centroids."""

    def __init__(self, linear: nn.Linear, n_clusters: int):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        weight_np = linear.weight.data.float().cpu().numpy().reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        kmeans.fit(weight_np)
        centroids = kmeans.cluster_centers_.squeeze()
        indices = kmeans.labels_.reshape(linear.weight.shape)
        self.register_buffer("centroids", torch.tensor(centroids, dtype=torch.float32))
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.bias = (
            nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        )

    def forward(self, x) -> Any:
        """Forward pass."""
        W = self.centroids[self.indices].to(x.dtype)
        out = F.linear(x, W, self.bias)
        return out


class KMeansCompressor(BaseCompressor):
    def __init__(self, model, n_clusters=256, **kwargs):
        super().__init__(model, name="kmeans", **kwargs)
        self.n_clusters = n_clusters

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                new_layer = KMeansLinear(module, self.n_clusters)
                parent = self._get_parent(name)
                if parent:
                    setattr(parent, name.split(".")[-1], new_layer)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()

    def _get_parent(self, full_name) -> Any:
        """ get parent."""
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
        return parent


@COMPRESSOR_REGISTRY.register("kmeans")
def kmeans_factory(model, comp_config) -> Any:
    """Kmeans factory."""
    n_clusters = getattr(comp_config, "kmeans_clusters", 256)
    compressor = KMeansCompressor(model, n_clusters=n_clusters)
    compressed_model, _ = compressor.compress()
    return compressed_model
