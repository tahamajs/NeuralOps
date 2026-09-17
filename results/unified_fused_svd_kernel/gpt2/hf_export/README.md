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
- **Validation Accuracy**: `70.07%`
- **F1 Score (Binary)**: `0.754`
- **F1 Macro**: `0.6859`
- **Precision**: `0.7379`
- **Recall**: `0.6969`
- **Validation Loss**: `0.5923`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.2635x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `98489088`
- **Trainable Recovery Params**: `1778688`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `464.46 MB`
- **Throughput**: `1955.65 tokens/sec`
- **Evaluation Time**: `97.56 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
