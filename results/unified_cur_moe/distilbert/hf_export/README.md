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
- **Validation Accuracy**: `73.39%`
- **F1 Score (Binary)**: `0.7717`
- **F1 Macro**: `0.7265`
- **Precision**: `0.756`
- **Recall**: `0.7312`
- **Validation Loss**: `0.5081`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.3806x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `48496398`
- **Trainable Recovery Params**: `514060`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `166.01 MB`
- **Throughput**: `44936.09 tokens/sec`
- **Evaluation Time**: `8.47 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
