---
tags:
- svd-linear-attention
- model-compression
- roberta
- pytorch
datasets:
- sst2
metrics:
- accuracy
- f1
---

# SVD Linear Attention / Compression: ROBERTA (roberta)

Validated factorized/adapted checkpoint of **roberta** using **roberta** evaluated on **SST-2**.

## Empirical Performance Results
- **Validation Accuracy**: `58.37%`
- **F1 Score (Binary)**: `0.6857`
- **F1 Macro**: `0.5347`
- **Precision**: `0.6294`
- **Recall**: `0.578`
- **Validation Loss**: `0.678`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9357x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `133208882`
- **Trainable Recovery Params**: `1771826`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `377.72 MB`
- **Throughput**: `41090.86 tokens/sec`
- **Evaluation Time**: `7.51 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
