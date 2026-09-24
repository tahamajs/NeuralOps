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
- **Validation Accuracy**: `88.99%`
- **F1 Score (Binary)**: `0.897`
- **F1 Macro**: `0.8894`
- **Precision**: `0.8944`
- **Recall**: `0.8889`
- **Validation Loss**: `0.3207`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0901x`
- **FLOPs Reduction**: `50.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `61422098`
- **Trainable Recovery Params**: `1544976`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `213.32 MB`
- **Throughput**: `14684.15 tokens/sec`
- **Evaluation Time**: `24.43 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
