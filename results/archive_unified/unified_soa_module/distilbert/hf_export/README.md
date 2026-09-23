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
- **Validation Accuracy**: `89.22%`
- **F1 Score (Binary)**: `0.898`
- **F1 Macro**: `0.8918`
- **Precision**: `0.895`
- **Recall**: `0.8914`
- **Validation Loss**: `0.3687`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1107x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `60282626`
- **Trainable Recovery Params**: `405504`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `293.12 MB`
- **Throughput**: `13809.54 tokens/sec`
- **Evaluation Time**: `23.12 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
