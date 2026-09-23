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
- **Validation Accuracy**: `62.39%`
- **F1 Score (Binary)**: `0.5941`
- **F1 Macro**: `0.6218`
- **Precision**: `0.6289`
- **Recall**: `0.6254`
- **Validation Loss**: `0.7741`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9905x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `67600274`
- **Trainable Recovery Params**: `645264`
- **Compressed / Injected Layers**: `24`
- **Peak VRAM**: `210.46 MB`
- **Throughput**: `5972.49 tokens/sec`
- **Evaluation Time**: `51.63 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
