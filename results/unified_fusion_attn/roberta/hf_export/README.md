---
tags: [svd-linear-attention, model-compression, fusion_attn, roberta, pytorch]
datasets: [sst2]
metrics: [accuracy, f1]
---
# SVD Linear Attention: ROBERTA (fusion_attn)
FUSIONATTN Multi-Paradigm Hybrid Sparse Attention Module — validated checkpoint evaluated on SST-2.
- **Accuracy**: `84.52%` | **Status**: `PASS` (threshold >= 56%)
- **F1**: `N/A` | **Peak VRAM**: `399.52 MB`
```python
import torch; checkpoint = torch.load("weights.pt", map_location="cpu")
```
Part of the SVD Linear Attention Benchmark Suite.
