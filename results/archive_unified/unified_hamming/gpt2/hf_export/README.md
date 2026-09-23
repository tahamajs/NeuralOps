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
- **Validation Accuracy**: `71.22%`
- **F1 Score (Binary)**: `0.7239`
- **F1 Macro**: `0.7116`
- **Precision**: `0.7125`
- **Recall**: `0.7116`
- **Validation Loss**: `0.5593`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9906x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `125621340`
- **Trainable Recovery Params**: `1179696`
- **Compressed / Injected Layers**: `48`
- **Peak VRAM**: `567.72 MB`
- **Throughput**: `11585.54 tokens/sec`
- **Evaluation Time**: `25.61 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
