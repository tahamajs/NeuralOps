---
tags:
- svd-linear-attention
- model-compression
- qwen3_5
- pytorch
datasets:
- sst2
metrics:
- accuracy
- f1
---

# SVD Linear Attention / Compression: QWEN3_5 (qwen3_5)

Validated factorized/adapted checkpoint of **qwen3_5** using **qwen3_5** evaluated on **SST-2**.

## Empirical Performance Results
- **Validation Accuracy**: `56.65%`
- **F1 Score (Binary)**: `0.6927`
- **F1 Macro**: `0.4786`
- **Precision**: `0.6663`
- **Recall**: `0.5592`
- **Validation Loss**: `0.6785`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9593x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `763217728`
- **Total Parameters After**: `795584368`
- **Trainable Recovery Params**: `11709488`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `1918.44 MB`
- **Throughput**: `4519.62 tokens/sec`
- **Evaluation Time**: `230.1 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
