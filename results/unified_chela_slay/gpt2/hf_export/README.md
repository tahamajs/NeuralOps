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
- **Validation Accuracy**: `85.89%`
- **F1 Score (Binary)**: `0.8712`
- **F1 Macro**: `0.8577`
- **Precision**: `0.8683`
- **Recall**: `0.8575`
- **Validation Loss**: `0.334`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9968x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `127206288`
- **Trainable Recovery Params**: `2655744`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `2291.25 MB`
- **Throughput**: `4127.17 tokens/sec`
- **Evaluation Time**: `198.1 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
