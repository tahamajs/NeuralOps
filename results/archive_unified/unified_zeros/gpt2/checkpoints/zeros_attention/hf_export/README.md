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
- **Validation Accuracy**: `59.75%`
- **F1 Score (Binary)**: `0.7023`
- **F1 Macro**: `0.5405`
- **Precision**: `0.6721`
- **Recall**: `0.5912`
- **Validation Loss**: `0.6691`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9906x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `125621040`
- **Trainable Recovery Params**: `1181232`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `1221.3 MB`
- **Throughput**: `9186.96 tokens/sec`
- **Evaluation Time**: `26.6 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
