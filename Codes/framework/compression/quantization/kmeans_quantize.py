# framework/compression/kmeans_quantize.py
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

class KMeansQuantize(nn.Module):
    def __init__(self, linear: nn.Linear, n_clusters: int):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        kmeans = KMeans(n_clusters=n_clusters)
        weight_np = linear.weight.data.cpu().numpy().reshape(-1, 1)
        kmeans.fit(weight_np)
        centroids = kmeans.cluster_centers_.squeeze()
        indices = kmeans.labels_.reshape(linear.weight.shape)
        self.register_buffer('centroids', torch.tensor(centroids))
        self.register_buffer('indices', torch.tensor(indices))
        self.bias = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.centroids[self.indices]
        return x @ W + (self.bias if self.bias is not None else 0)