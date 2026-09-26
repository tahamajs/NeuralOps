---
tags:
- svd-linear-attention
- model-compression
- gpt2
- pytorch
datasets:
- sst2
metrics:
- accuracy
- f1
---

# SVD Linear Attention / Compression: GPT2 (gpt2)

Validated factorized/adapted checkpoint of **gpt2** using **gpt2** evaluated on **SST-2**.

## Empirical Performance Results
- **Validation Accuracy**: `59.52%`
- **F1 Score (Binary)**: `0.6735`
- **F1 Macro**: `0.5705`
- **Precision**: `0.6155`
- **Recall**: `0.591`
- **Validation Loss**: `0.6649`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.2251x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `101579088`
- **Trainable Recovery Params**: `3090000`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `476.0 MB`
- **Throughput**: `8474.53 tokens/sec`
- **Evaluation Time**: `38.79 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
