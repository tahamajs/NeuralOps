"""KMeans‑based weight clustering compressor. Registered as 'kmeans'."""

import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

class KMeansLinear(nn.Module):
    """Linear layer with weights replaced by k‑means centroids."""
    def __init__(self, linear: nn.Linear, n_clusters: int):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        weight_np = linear.weight.data.float().cpu().numpy().reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(weight_np)
        centroids = kmeans.cluster_centers_.squeeze()
        indices = kmeans.labels_.reshape(linear.weight.shape)
        self.register_buffer('centroids', torch.tensor(centroids, dtype=torch.float32))
        self.register_buffer('indices', torch.tensor(indices, dtype=torch.long))
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None

    def forward(self, x):
        W = self.centroids[self.indices].to(x.dtype)
        out = F.linear(x, W, self.bias)
        return out

@COMPRESSOR_REGISTRY.register("kmeans")
class KMeansCompressor(BaseCompressor):
    def __init__(self, model, n_clusters=256, **kwargs):
        super().__init__(model, name="kmeans", **kwargs)
        self.n_clusters = n_clusters

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                new_layer = KMeansLinear(module, self.n_clusters)
                parent = self._get_parent(name)
                if parent:
                    setattr(parent, name.split('.')[-1], new_layer)
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()

    def _get_parent(self, full_name):
        parts = full_name.split('.')
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
        return parent