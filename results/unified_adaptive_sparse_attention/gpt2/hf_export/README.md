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
- **Validation Accuracy**: `75.11%`
- **F1 Score (Binary)**: `0.7797`
- **F1 Macro**: `0.7469`
- **Precision**: `0.7643`
- **Recall**: `0.749`
- **Validation Loss**: `1.2666`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `127095708`
- **Trainable Recovery Params**: `2654364`
- **Compressed / Injected Layers**: `12`
- **Peak VRAM**: `443.2 MB`
- **Throughput**: `7902.69 tokens/sec`
- **Evaluation Time**: `35.55 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
