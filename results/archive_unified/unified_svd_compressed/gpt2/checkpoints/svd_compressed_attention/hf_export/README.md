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
- **Validation Accuracy**: `76.15%`
- **F1 Score (Binary)**: `0.7965`
- **F1 Macro**: `0.7542`
- **Precision**: `0.7892`
- **Recall**: `0.7586`
- **Validation Loss**: `0.4723`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1717x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `108221196`
- **Trainable Recovery Params**: `1474572`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `300.29 MB`
- **Throughput**: `10244.56 tokens/sec`
- **Evaluation Time**: `34.77 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
