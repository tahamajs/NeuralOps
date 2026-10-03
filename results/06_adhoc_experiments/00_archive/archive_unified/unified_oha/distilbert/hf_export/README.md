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
- **Validation Accuracy**: `90.48%`
- **F1 Score (Binary)**: `0.9073`
- **F1 Macro**: `0.9048`
- **Precision**: `0.905`
- **Recall**: `0.9046`
- **Validation Loss**: `0.305`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67895138`
- **Trainable Recovery Params**: `645216`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `208.69 MB`
- **Throughput**: `17434.89 tokens/sec`
- **Evaluation Time**: `16.31 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
