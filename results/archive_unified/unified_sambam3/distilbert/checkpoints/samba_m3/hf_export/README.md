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
- **Validation Accuracy**: `89.45%`
- **F1 Score (Binary)**: `0.8991`
- **F1 Macro**: `0.8943`
- **Precision**: `0.896`
- **Recall**: `0.894`
- **Validation Loss**: `0.4664`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.8132x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `82331930`
- **Trainable Recovery Params**: `1698074`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `425.24 MB`
- **Throughput**: `31470.26 tokens/sec`
- **Evaluation Time**: `10.18 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
