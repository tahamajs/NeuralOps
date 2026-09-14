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
- **Validation Accuracy**: `78.44%`
- **F1 Score (Binary)**: `0.8216`
- **F1 Macro**: `0.7746`
- **Precision**: `0.8339`
- **Recall**: `0.7808`
- **Validation Loss**: `0.686`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9884x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67741538`
- **Trainable Recovery Params**: `786528`
- **Compressed / Injected Layers**: `24`
- **Peak VRAM**: `208.28 MB`
- **Throughput**: `13987.21 tokens/sec`
- **Evaluation Time**: `24.99 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
