# Tutorial 3: Model Compression Pipeline

Compress large models using Low-Rank SVD, Pruning, and Quantization engines.

```python
import torch
import torch.nn as nn
from neuralops.core.registry import COMPRESSOR_REGISTRY
import neuralops.compression_engine

# Build a sample neural network
net = nn.Sequential(
    nn.Linear(256, 128),
    nn.GELU(),
    nn.Linear(128, 10)
)

# Apply Direct SVD Compressor
svd_cls = COMPRESSOR_REGISTRY["direct_svd"]
compressor = svd_cls(net, rank=32)
compressed_net = compressor.compress()

print("Original param count:", sum(p.numel() for p in net.parameters()))
print("Compressed param count:", sum(p.numel() for p in compressed_net.parameters()))
```
