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
- **Validation Accuracy**: `90.25%`
- **F1 Score (Binary)**: `0.9063`
- **F1 Macro**: `0.9024`
- **Precision**: `0.9035`
- **Recall**: `0.9021`
- **Validation Loss**: `0.7666`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9884x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67741472`
- **Trainable Recovery Params**: `1378592`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `331.47 MB`
- **Throughput**: `75937.59 tokens/sec`
- **Evaluation Time**: `4.2 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
