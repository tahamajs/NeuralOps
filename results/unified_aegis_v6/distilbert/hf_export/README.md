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
- **Validation Accuracy**: `89.68%`
- **F1 Score (Binary)**: `0.9017`
- **F1 Macro**: `0.8965`
- **Precision**: `0.8987`
- **Recall**: `0.8962`
- **Validation Loss**: `0.2853`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1062x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `60528464`
- **Trainable Recovery Params**: `645576`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `225.11 MB`
- **Throughput**: `6941.52 tokens/sec`
- **Evaluation Time**: `39.65 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
