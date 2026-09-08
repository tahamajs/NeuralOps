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
- **Validation Accuracy**: `74.31%`
- **F1 Score (Binary)**: `0.7068`
- **F1 Macro**: `0.7391`
- **Precision**: `0.7643`
- **Recall**: `0.7456`
- **Validation Loss**: `0.5495`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9977x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `127095852`
- **Trainable Recovery Params**: `2656044`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `340.1 MB`
- **Throughput**: `50053.2 tokens/sec`
- **Evaluation Time**: `6.51 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
