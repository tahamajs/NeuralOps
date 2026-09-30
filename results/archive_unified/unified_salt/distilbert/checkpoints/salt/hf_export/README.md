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
- **Validation Accuracy**: `79.01%`
- **F1 Score (Binary)**: `0.8017`
- **F1 Macro**: `0.7894`
- **Precision**: `0.7921`
- **Recall**: `0.7893`
- **Validation Loss**: `0.4943`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.2404x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `53979674`
- **Trainable Recovery Params**: `1198872`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `334.38 MB`
- **Throughput**: `26469.35 tokens/sec`
- **Evaluation Time**: `22.56 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
