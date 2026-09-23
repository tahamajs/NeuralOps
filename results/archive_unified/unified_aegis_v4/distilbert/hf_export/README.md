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
- **Validation Accuracy**: `84.75%`
- **F1 Score (Binary)**: `0.8537`
- **F1 Macro**: `0.8472`
- **Precision**: `0.8484`
- **Recall**: `0.847`
- **Validation Loss**: `0.5132`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9884x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67741538`
- **Trainable Recovery Params**: `1378658`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `347.69 MB`
- **Throughput**: `53320.93 tokens/sec`
- **Evaluation Time**: `10.2 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
