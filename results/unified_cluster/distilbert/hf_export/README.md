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
- **Validation Accuracy**: `90.02%`
- **F1 Score (Binary)**: `0.9039`
- **F1 Macro**: `0.9001`
- **Precision**: `0.901`
- **Recall**: `0.8998`
- **Validation Loss**: `0.3342`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9913x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67544852`
- **Trainable Recovery Params**: `589830`
- **Compressed / Injected Layers**: `24`
- **Peak VRAM**: `331.87 MB`
- **Throughput**: `3165.73 tokens/sec`
- **Evaluation Time**: `65.23 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
