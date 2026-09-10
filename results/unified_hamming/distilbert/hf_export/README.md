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
- **Validation Accuracy**: `75.92%`
- **F1 Score (Binary)**: `0.7965`
- **F1 Macro**: `0.7508`
- **Precision**: `0.7914`
- **Recall**: `0.7561`
- **Validation Loss**: `0.7328`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9913x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67544984`
- **Trainable Recovery Params**: `1182104`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `331.45 MB`
- **Throughput**: `93288.52 tokens/sec`
- **Evaluation Time**: `3.29 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
