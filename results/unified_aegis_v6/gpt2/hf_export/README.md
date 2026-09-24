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
- **Validation Accuracy**: `78.67%`
- **F1 Score (Binary)**: `0.7609`
- **F1 Macro**: `0.7842`
- **Precision**: `0.8056`
- **Recall**: `0.7889`
- **Validation Loss**: `0.4409`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1152x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `111588252`
- **Trainable Recovery Params**: `1291152`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `1110.17 MB`
- **Throughput**: `2556.58 tokens/sec`
- **Evaluation Time**: `114.21 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
