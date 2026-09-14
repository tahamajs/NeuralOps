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
- **Validation Accuracy**: `59.98%`
- **F1 Score (Binary)**: `0.7142`
- **F1 Macro**: `0.5234`
- **Precision**: `0.7385`
- **Recall**: `0.5926`
- **Validation Loss**: `0.6305`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.2491x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `101512020`
- **Trainable Recovery Params**: `2995248`
- **Compressed / Injected Layers**: `24`
- **Peak VRAM**: `470.37 MB`
- **Throughput**: `5818.13 tokens/sec`
- **Evaluation Time**: `54.5 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
