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
- **Validation Accuracy**: `79.24%`
- **F1 Score (Binary)**: `0.7964`
- **F1 Macro**: `0.7924`
- **Precision**: `0.7924`
- **Recall**: `0.7923`
- **Validation Loss**: `0.4875`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9814x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `126800688`
- **Trainable Recovery Params**: `2359344`
- **Compressed / Injected Layers**: `48`
- **Peak VRAM**: `571.18 MB`
- **Throughput**: `12089.63 tokens/sec`
- **Evaluation Time**: `14.87 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
