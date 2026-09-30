---
tags:
- svd-linear-attention
- model-compression
- distilbert
- pytorch
datasets:
- sst2
metrics:
- accuracy
- f1
---

# SVD Linear Attention / Compression: DISTILBERT (distilbert)

Validated factorized/adapted checkpoint of **distilbert** using **distilbert** evaluated on **SST-2**.

## Empirical Performance Results
- **Validation Accuracy**: `90.71%`
- **F1 Score (Binary)**: `0.9111`
- **F1 Macro**: `0.9069`
- **Precision**: `0.9085`
- **Recall**: `0.9066`
- **Validation Loss**: `0.6593`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.8564x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `78184748`
- **Trainable Recovery Params**: `1186604`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `378.15 MB`
- **Throughput**: `66788.52 tokens/sec`
- **Evaluation Time**: `4.78 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
