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
- **Validation Accuracy**: `58.83%`
- **F1 Score (Binary)**: `0.6913`
- **F1 Macro**: `0.5367`
- **Precision**: `0.6423`
- **Recall**: `0.5824`
- **Validation Loss**: `0.6673`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.2439x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `100037460`
- **Trainable Recovery Params**: `1520688`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `643.49 MB`
- **Throughput**: `5217.46 tokens/sec`
- **Evaluation Time**: `55.53 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
