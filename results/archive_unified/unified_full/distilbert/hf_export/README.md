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
- **F1 Score (Binary)**: `0.9099`
- **F1 Macro**: `0.907`
- **Precision**: `0.9075`
- **Recall**: `0.9068`
- **Validation Loss**: `0.3027`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `66955010`
- **Trainable Recovery Params**: `0`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `388.01 MB`
- **Throughput**: `53818.36 tokens/sec`
- **Evaluation Time**: `2.91 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
