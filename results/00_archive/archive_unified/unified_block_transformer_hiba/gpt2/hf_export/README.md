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
- **Validation Accuracy**: `59.98%`
- **F1 Score (Binary)**: `0.6947`
- **F1 Macro**: `0.557`
- **Precision**: `0.6481`
- **Recall**: `0.5943`
- **Validation Loss**: `0.6664`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9906x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `125621040`
- **Trainable Recovery Params**: `1181232`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `568.71 MB`
- **Throughput**: `47157.13 tokens/sec`
- **Evaluation Time**: `7.42 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
