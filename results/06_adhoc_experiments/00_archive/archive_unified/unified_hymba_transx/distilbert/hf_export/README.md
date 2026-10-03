---
tags: [svd-linear-attention, model-compression, hymba_transx, distilbert, pytorch]
datasets: [sst2]
metrics: [accuracy, f1]
---
# SVD Linear Attention: DISTILBERT (hymba_transx)
HymbaTransX: Hybrid Differential Transformer & State Space Model Block — validated checkpoint evaluated on SST-2.
- **Accuracy**: `90.48%` | **Status**: `PASS` (threshold >= 56%)
- **F1**: `N/A` | **Peak VRAM**: `389.51 MB`
```python
import torch; checkpoint = torch.load("weights.pt", map_location="cpu")
```
Part of the SVD Linear Attention Benchmark Suite.
