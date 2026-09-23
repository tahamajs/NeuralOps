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
- **Validation Accuracy**: `81.88%`
- **F1 Score (Binary)**: `0.826`
- **F1 Macro**: `0.8185`
- **Precision**: `0.8195`
- **Recall**: `0.8183`
- **Validation Loss**: `0.4785`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.227x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `66955010`
- **Total Parameters After**: `54569480`
- **Trainable Recovery Params**: `2380808`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `288.53 MB`
- **Throughput**: `59614.2 tokens/sec`
- **Evaluation Time**: `6.36 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
