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
- **Validation Accuracy**: `60.89%`
- **F1 Score (Binary)**: `0.7058`
- **F1 Macro**: `0.5614`
- **Precision**: `0.6745`
- **Recall**: `0.6031`
- **Validation Loss**: `0.6339`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.986x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `126211008`
- **Trainable Recovery Params**: `1771200`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `573.78 MB`
- **Throughput**: `33657.55 tokens/sec`
- **Evaluation Time**: `8.83 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
