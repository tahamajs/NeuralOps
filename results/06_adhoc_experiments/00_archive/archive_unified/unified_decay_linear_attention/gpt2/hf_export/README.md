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
- **Validation Accuracy**: `63.19%`
- **F1 Score (Binary)**: `0.7106`
- **F1 Macro**: `0.6025`
- **Precision**: `0.6755`
- **Recall**: `0.6271`
- **Validation Loss**: `0.6273`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9897x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `125731740`
- **Trainable Recovery Params**: `1291932`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `570.16 MB`
- **Throughput**: `36650.11 tokens/sec`
- **Evaluation Time**: `8.86 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
