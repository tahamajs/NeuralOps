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
- **Validation Accuracy**: `81.65%`
- **F1 Score (Binary)**: `0.8403`
- **F1 Macro**: `0.8123`
- **Precision**: `0.8406`
- **Recall**: `0.8141`
- **Validation Loss**: `0.406`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9946x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `127488960`
- **Trainable Recovery Params**: `3047616`
- **Compressed / Injected Layers**: `72`
- **Peak VRAM**: `339.09 MB`
- **Throughput**: `9456.01 tokens/sec`
- **Evaluation Time**: `39.58 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
