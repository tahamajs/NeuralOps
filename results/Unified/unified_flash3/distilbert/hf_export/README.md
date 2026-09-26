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
- **F1 Score (Binary)**: `0.9101`
- **F1 Macro**: `0.907`
- **Precision**: `0.9076`
- **Recall**: `0.9068`
- **Validation Loss**: `0.2584`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67544834`
- **Trainable Recovery Params**: `589824`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `234.33 MB`
- **Throughput**: `18480.11 tokens/sec`
- **Evaluation Time**: `14.36 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
