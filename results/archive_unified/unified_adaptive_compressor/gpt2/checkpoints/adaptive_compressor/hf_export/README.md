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
- **Validation Accuracy**: `76.95%`
- **F1 Score (Binary)**: `0.8016`
- **F1 Macro**: `0.7633`
- **Precision**: `0.7941`
- **Recall**: `0.7668`
- **Validation Loss**: `0.4773`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0021x`
- **FLOPs Reduction**: `44.79%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `124177153`
- **Trainable Recovery Params**: `326401`
- **Compressed / Injected Layers**: `1`
- **Peak VRAM**: `638.52 MB`
- **Throughput**: `22313.99 tokens/sec`
- **Evaluation Time**: `16.81 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
