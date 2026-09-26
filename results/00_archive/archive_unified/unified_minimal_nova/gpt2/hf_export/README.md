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
- **Validation Accuracy**: `58.83%`
- **F1 Score (Binary)**: `0.705`
- **F1 Macro**: `0.5119`
- **Precision**: `0.7017`
- **Recall**: `0.5812`
- **Validation Loss**: `0.6583`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1658x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `106746624`
- **Trainable Recovery Params**: `1179648`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `494.09 MB`
- **Throughput**: `11601.55 tokens/sec`
- **Evaluation Time**: `29.11 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
