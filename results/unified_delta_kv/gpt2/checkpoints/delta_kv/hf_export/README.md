---
tags:
- svd-linear-attention
- model-compression
- delta-kv
- gpt2
- pytorch
datasets:
- sst2
metrics:
- accuracy
- f1
---

# DeltaKV Compression: GPT2 (gpt2)

Residual-based KV cache compression via long-range similarity applied to **gpt2** evaluated on **SST-2**.

## Empirical Performance Results
- **Validation Accuracy**: `87.39%`
- **F1 Score**: `0.8706`
- **Validation Loss**: `0.354`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0x`
- **Compressed Layers**: `5`
- **Peak VRAM**: `384.13 MB`
- **Throughput**: `14405.64 tokens/sec`

## Usage
```python
import torch

checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded DeltaKV compressed checkpoint successfully.")
```
