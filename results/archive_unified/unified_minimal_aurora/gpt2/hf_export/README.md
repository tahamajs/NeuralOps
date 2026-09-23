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
- **Validation Accuracy**: `59.86%`
- **F1 Score (Binary)**: `0.7107`
- **F1 Macro**: `0.5277`
- **Precision**: `0.7146`
- **Recall**: `0.5917`
- **Validation Loss**: `0.6468`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1658x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `106746648`
- **Trainable Recovery Params**: `1179672`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `495.67 MB`
- **Throughput**: `11208.84 tokens/sec`
- **Evaluation Time**: `31.82 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
