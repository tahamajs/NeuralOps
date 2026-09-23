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
- **Validation Accuracy**: `81.08%`
- **F1 Score (Binary)**: `0.7854`
- **F1 Macro**: `0.8081`
- **Precision**: `0.8348`
- **Recall**: `0.8132`
- **Validation Loss**: `0.7658`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.06x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `763217728`
- **Total Parameters After**: `720025408`
- **Trainable Recovery Params**: `10382336`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `1767.71 MB`
- **Throughput**: `1100.17 tokens/sec`
- **Evaluation Time**: `366.55 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
