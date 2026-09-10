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
- **Validation Accuracy**: `80.28%`
- **F1 Score (Binary)**: `0.8097`
- **F1 Macro**: `0.8025`
- **Precision**: `0.8032`
- **Recall**: `0.8023`
- **Validation Loss**: `0.538`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9913x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67544858`
- **Trainable Recovery Params**: `1181978`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `341.15 MB`
- **Throughput**: `48706.63 tokens/sec`
- **Evaluation Time**: `5.71 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
