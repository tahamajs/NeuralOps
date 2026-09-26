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
- **Validation Accuracy**: `87.04%`
- **F1 Score (Binary)**: `0.8791`
- **F1 Macro**: `0.8697`
- **Precision**: `0.8752`
- **Recall**: `0.8694`
- **Validation Loss**: `0.336`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.2527x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `53449064`
- **Trainable Recovery Params**: `668262`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `301.26 MB`
- **Throughput**: `11663.21 tokens/sec`
- **Evaluation Time**: `27.51 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
