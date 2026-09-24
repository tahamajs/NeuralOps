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
- **Validation Accuracy**: `90.83%`
- **F1 Score (Binary)**: `0.913`
- **F1 Macro**: `0.908`
- **Precision**: `0.9109`
- **Recall**: `0.9076`
- **Validation Loss**: `0.3414`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67504866`
- **Trainable Recovery Params**: `589824`
- **Compressed / Injected Layers**: `6`
- **Peak VRAM**: `215.4 MB`
- **Throughput**: `20319.54 tokens/sec`
- **Evaluation Time**: `19.31 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
