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
- **Validation Accuracy**: `78.10%`
- **F1 Score (Binary)**: `0.7885`
- **F1 Macro**: `0.7807`
- **Precision**: `0.7813`
- **Recall**: `0.7806`
- **Validation Loss**: `0.4792`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9913x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67544840`
- **Trainable Recovery Params**: `1181960`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `336.97 MB`
- **Throughput**: `18046.26 tokens/sec`
- **Evaluation Time**: `15.26 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
