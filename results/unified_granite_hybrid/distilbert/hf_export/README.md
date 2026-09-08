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
- **Validation Accuracy**: `63.53%`
- **F1 Score (Binary)**: `0.7319`
- **F1 Macro**: `0.581`
- **Precision**: `0.754`
- **Recall**: `0.6289`
- **Validation Loss**: `0.5722`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.3092x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `216526106`
- **Trainable Recovery Params**: `1338626`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `1605.33 MB`
- **Throughput**: `12421.3 tokens/sec`
- **Evaluation Time**: `24.13 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
