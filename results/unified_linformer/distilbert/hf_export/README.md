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
- **Validation Accuracy**: `90.83%`
- **F1 Score (Binary)**: `0.9119`
- **F1 Macro**: `0.9081`
- **Precision**: `0.9094`
- **Recall**: `0.9078`
- **Validation Loss**: `0.6932`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9855x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67938056`
- **Trainable Recovery Params**: `1575176`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `331.96 MB`
- **Throughput**: `74868.53 tokens/sec`
- **Evaluation Time**: `4.13 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
