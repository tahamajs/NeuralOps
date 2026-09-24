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
- **Validation Accuracy**: `71.44%`
- **F1 Score (Binary)**: `0.7734`
- **F1 Macro**: `0.6937`
- **Precision**: `0.7806`
- **Recall**: `0.7099`
- **Validation Loss**: `0.5148`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1193x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `111176448`
- **Trainable Recovery Params**: `884736`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `377.72 MB`
- **Throughput**: `12160.68 tokens/sec`
- **Evaluation Time**: `26.88 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
