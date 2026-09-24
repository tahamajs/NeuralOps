---
tags: [svd-linear-attention, model-compression, galerkin_attention, gpt2, pytorch]
datasets: [sst2]
metrics: [accuracy, f1]
---
# SVD Linear Attention: GPT2 (galerkin_attention)
Galerkin Attention: Petrov-Galerkin Projection Attention — validated checkpoint evaluated on SST-2.
- **Accuracy**: `88.88%` | **Status**: `PASS` (threshold >= 56%)
- **F1**: `N/A` | **Peak VRAM**: `430.51 MB`
```python
import torch; checkpoint = torch.load("weights.pt", map_location="cpu")
```
Part of the SVD Linear Attention Benchmark Suite.
