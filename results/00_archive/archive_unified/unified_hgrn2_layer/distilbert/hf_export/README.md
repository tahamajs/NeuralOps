---
tags: [svd-linear-attention, model-compression, hgrn2_layer, distilbert, pytorch]
datasets: [sst2]
metrics: [accuracy, f1]
---
# SVD Linear Attention: DISTILBERT (hgrn2_layer)
Hierarchically Gated Recurrent Network V2 (HGRN-2) Layer — validated checkpoint evaluated on SST-2.
- **Accuracy**: `62.04%` | **Status**: `PASS` (threshold >= 56%)
- **F1**: `N/A` | **Peak VRAM**: `352.34 MB`
```python
import torch; checkpoint = torch.load("weights.pt", map_location="cpu")
```
Part of the SVD Linear Attention Benchmark Suite.
