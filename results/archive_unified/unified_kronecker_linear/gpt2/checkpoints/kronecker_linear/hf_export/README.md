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
- **Validation Accuracy**: `79.36%`
- **F1 Score (Binary)**: `0.7978`
- **F1 Macro**: `0.7935`
- **Precision**: `0.7935`
- **Recall**: `0.7935`
- **Validation Loss**: `0.4875`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `124441344`
- **Trainable Recovery Params**: `0`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `628.41 MB`
- **Throughput**: `26580.07 tokens/sec`
- **Evaluation Time**: `13.04 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
