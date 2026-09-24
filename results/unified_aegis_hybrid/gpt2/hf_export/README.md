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
- **Validation Accuracy**: `65.71%`
- **F1 Score (Binary)**: `0.7366`
- **F1 Macro**: `0.6228`
- **Precision**: `0.7306`
- **Recall**: `0.6518`
- **Validation Loss**: `0.5957`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.211x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `102758688`
- **Trainable Recovery Params**: `4269600`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `775.45 MB`
- **Throughput**: `3306.96 tokens/sec`
- **Evaluation Time**: `76.92 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
