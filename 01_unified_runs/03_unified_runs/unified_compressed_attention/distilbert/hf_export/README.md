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
- **Validation Accuracy**: `89.11%`
- **F1 Score (Binary)**: `0.8975`
- **F1 Macro**: `0.8906`
- **Precision**: `0.8947`
- **Recall**: `0.8902`
- **Validation Loss**: `0.3267`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1099x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `60322562`
- **Trainable Recovery Params**: `442368`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `234.2 MB`
- **Throughput**: `27425.93 tokens/sec`
- **Evaluation Time**: `11.55 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
