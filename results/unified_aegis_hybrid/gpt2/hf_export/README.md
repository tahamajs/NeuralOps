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
- **F1 Score (Binary)**: `0.7622`
- **F1 Macro**: `0.7025`
- **Precision**: `0.7472`
- **Recall**: `0.711`
- **Validation Loss**: `0.5343`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.2304x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `103053648`
- **Trainable Recovery Params**: `4564560`
- **Compressed / Injected Layers**: `24`
- **Peak VRAM**: `300.71 MB`
- **Throughput**: `9822.11 tokens/sec`
- **Evaluation Time**: `59.64 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
