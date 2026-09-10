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
- **Validation Accuracy**: `84.29%`
- **F1 Score (Binary)**: `0.8541`
- **F1 Macro**: `0.842`
- **Precision**: `0.848`
- **Recall**: `0.8418`
- **Validation Loss**: `0.7081`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9884x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67741538`
- **Trainable Recovery Params**: `1378658`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `336.59 MB`
- **Throughput**: `53971.51 tokens/sec`
- **Evaluation Time**: `10.12 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
