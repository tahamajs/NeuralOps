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
- **F1 Score (Binary)**: `0.8207`
- **F1 Macro**: `0.7887`
- **Precision**: `0.8166`
- **Recall**: `0.7911`
- **Validation Loss**: `0.4496`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1717x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `108221184`
- **Trainable Recovery Params**: `2654208`
- **Compressed / Injected Layers**: `84`
- **Peak VRAM**: `319.87 MB`
- **Throughput**: `9739.09 tokens/sec`
- **Evaluation Time**: `75.6 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
