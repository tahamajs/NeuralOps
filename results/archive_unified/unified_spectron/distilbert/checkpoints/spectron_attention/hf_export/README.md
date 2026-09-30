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
- **Validation Accuracy**: `86.58%`
- **F1 Score (Binary)**: `0.8765`
- **F1 Macro**: `0.8648`
- **Precision**: `0.8732`
- **Recall**: `0.8645`
- **Validation Loss**: `0.4168`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1523x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `58107656`
- **Trainable Recovery Params**: `610568`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `294.06 MB`
- **Throughput**: `73633.04 tokens/sec`
- **Evaluation Time**: `6.7 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
