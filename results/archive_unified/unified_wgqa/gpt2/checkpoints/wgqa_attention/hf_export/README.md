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
- **Validation Accuracy**: `87.16%`
- **F1 Score (Binary)**: `0.879`
- **F1 Macro**: `0.8711`
- **Precision**: `0.8748`
- **Recall**: `0.8707`
- **Validation Loss**: `0.3123`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9977x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `127095888`
- **Trainable Recovery Params**: `2654544`
- **Compressed / Injected Layers**: `72`
- **Peak VRAM**: `343.63 MB`
- **Throughput**: `8935.39 tokens/sec`
- **Evaluation Time**: `31.24 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
