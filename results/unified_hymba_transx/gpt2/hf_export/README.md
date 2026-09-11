---
tags: [svd-linear-attention, model-compression, hymba_transx, gpt2, pytorch]
datasets: [sst2]
metrics: [accuracy, f1]
---
# SVD Linear Attention: GPT2 (hymba_transx)
HymbaTransX: Hybrid Differential Transformer & State Space Model Block — validated checkpoint evaluated on SST-2.
- **Accuracy**: `57.57%` | **Status**: `PASS` (threshold >= 56%)
- **F1**: `N/A` | **Peak VRAM**: `803.95 MB`
```python
import torch; checkpoint = torch.load("weights.pt", map_location="cpu")
```
Part of the SVD Linear Attention Benchmark Suite.
