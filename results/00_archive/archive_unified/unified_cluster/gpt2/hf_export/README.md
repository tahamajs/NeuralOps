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
- **Validation Accuracy**: `87.96%`
- **F1 Score (Binary)**: `0.885`
- **F1 Macro**: `0.8793`
- **Precision**: `0.8811`
- **Recall**: `0.879`
- **Validation Loss**: `0.3006`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9977x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `127095588`
- **Trainable Recovery Params**: `2654220`
- **Compressed / Injected Layers**: `72`
- **Peak VRAM**: `397.23 MB`
- **Throughput**: `10707.08 tokens/sec`
- **Evaluation Time**: `27.24 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
