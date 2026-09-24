---
tags: [svd-linear-attention, model-compression, hgrn2_layer, gpt2, pytorch]
datasets: [sst2]
metrics: [accuracy, f1]
---
# SVD Linear Attention: GPT2 (hgrn2_layer)
Hierarchically Gated Recurrent Network V2 (HGRN-2) Layer - validated checkpoint evaluated on SST-2.
- **Accuracy**: `89.33%` | **Status**: `PASS` (threshold >= 56%)
- **F1**: `N/A` | **Peak VRAM**: `382.13 MB`
```python
import torch; checkpoint = torch.load("weights.pt", map_location="cpu")
```
Part of the SVD Linear Attention Benchmark Suite.
