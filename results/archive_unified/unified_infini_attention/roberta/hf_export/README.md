---
tags: [svd-linear-attention, model-compression, infini_attention, roberta, pytorch]
datasets: [sst2]
metrics: [accuracy, f1]
---
# SVD Linear Attention: ROBERTA (infini_attention)
Infini-Attention with Compressive Memory and Sliding Window — validated checkpoint evaluated on SST-2.
- **Accuracy**: `56.54%` | **Status**: `PASS` (threshold >= 56%)
- **F1**: `N/A` | **Peak VRAM**: `406.54 MB`
```python
import torch; checkpoint = torch.load("weights.pt", map_location="cpu")
```
Part of the SVD Linear Attention Benchmark Suite.
