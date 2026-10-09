# Tutorial 1: Quickstart with NeuralOps

NeuralOps provides 159 attention mechanisms and 73 model compression engines for PyTorch.

```python
import torch
import neuralops
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
import neuralops.attentions, neuralops.compression_engine

print(f"NeuralOps Version: {neuralops.__version__}")
print(f"Registered Attentions: {len(ATTENTION_REGISTRY)}")
print(f"Registered Compressors: {len(COMPRESSOR_REGISTRY)}")

# Instantiate a Cosformer attention module
cosformer_cls = ATTENTION_REGISTRY["cosformer"]
attn = cosformer_cls()

# Forward pass with (batch=2, seq_len=16, dim=64)
x = torch.randn(2, 16, 64)
out = attn(x)
if isinstance(out, tuple):
    out = out[0]
print("Cosformer output shape:", out.shape)
```
