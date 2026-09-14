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
- **Validation Accuracy**: `68.35%`
- **F1 Score (Binary)**: `0.7583`
- **F1 Macro**: `0.6499`
- **Precision**: `0.7786`
- **Recall**: `0.678`
- **Validation Loss**: `0.5852`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1717x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `108221184`
- **Trainable Recovery Params**: `2654208`
- **Compressed / Injected Layers**: `24`
- **Peak VRAM**: `301.18 MB`
- **Throughput**: `11023.39 tokens/sec`
- **Evaluation Time**: `35.23 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
