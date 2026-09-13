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
- **F1 Score (Binary)**: `0.903`
- **F1 Macro**: `0.9001`
- **Precision**: `0.9005`
- **Recall**: `0.9`
- **Validation Loss**: `0.4359`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.6075x`
- **FLOPs Reduction**: `62.5%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `110205728`
- **Trainable Recovery Params**: `1200416`
- **Compressed / Injected Layers**: `96`
- **Peak VRAM**: `554.24 MB`
- **Throughput**: `31613.91 tokens/sec`
- **Evaluation Time**: `10.29 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
