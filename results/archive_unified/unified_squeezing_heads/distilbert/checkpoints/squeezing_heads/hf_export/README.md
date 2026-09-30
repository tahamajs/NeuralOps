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
- **Validation Accuracy**: `90.60%`
- **F1 Score (Binary)**: `0.9085`
- **F1 Macro**: `0.9059`
- **Precision**: `0.9062`
- **Recall**: `0.9058`
- **Validation Loss**: `0.3656`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9913x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67544834`
- **Trainable Recovery Params**: `589824`
- **Compressed / Injected Layers**: `24`
- **Peak VRAM**: `196.03 MB`
- **Throughput**: `27497.71 tokens/sec`
- **Evaluation Time**: `10.98 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
