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
- **Validation Accuracy**: `57.80%`
- **F1 Score (Binary)**: `0.7042`
- **F1 Macro**: `0.4841`
- **Precision**: `0.7321`
- **Recall**: `0.5703`
- **Validation Loss**: `1.0463`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.2635x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `98489088`
- **Trainable Recovery Params**: `1778688`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `330.56 MB`
- **Throughput**: `1498.37 tokens/sec`
- **Evaluation Time**: `129.16 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
