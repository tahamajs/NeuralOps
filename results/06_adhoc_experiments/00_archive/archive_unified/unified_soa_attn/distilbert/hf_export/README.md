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
- **Validation Accuracy**: `81.31%`
- **F1 Score (Binary)**: `0.8191`
- **F1 Macro**: `0.8129`
- **Precision**: `0.8134`
- **Recall**: `0.8127`
- **Validation Loss**: `0.5353`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1671x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `57370376`
- **Trainable Recovery Params**: `739592`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `282.69 MB`
- **Throughput**: `86194.93 tokens/sec`
- **Evaluation Time**: `5.21 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
