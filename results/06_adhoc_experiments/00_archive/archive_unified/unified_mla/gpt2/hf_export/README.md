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
- **Validation Accuracy**: `57.22%`
- **F1 Score (Binary)**: `0.526`
- **F1 Macro**: `0.5681`
- **Precision**: `0.5777`
- **Recall**: `0.5742`
- **Validation Loss**: `0.6801`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1137x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `111741708`
- **Trainable Recovery Params**: `591372`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `511.85 MB`
- **Throughput**: `45789.91 tokens/sec`
- **Evaluation Time**: `8.37 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
