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
- **Validation Accuracy**: `76.38%`
- **F1 Score (Binary)**: `0.8008`
- **F1 Macro**: `0.7553`
- **Precision**: `0.7977`
- **Recall**: `0.7606`
- **Validation Loss**: `0.4834`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `125621292`
- **Trainable Recovery Params**: `1179948`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `429.55 MB`
- **Throughput**: `1276.18 tokens/sec`
- **Evaluation Time**: `206.53 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
