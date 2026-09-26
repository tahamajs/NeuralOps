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
- **Validation Accuracy**: `58.49%`
- **F1 Score (Binary)**: `0.6953`
- **F1 Macro**: `0.5221`
- **Precision**: `0.6565`
- **Recall**: `0.5784`
- **Validation Loss**: `0.6581`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9906x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `125621148`
- **Trainable Recovery Params**: `1181340`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `567.69 MB`
- **Throughput**: `42855.01 tokens/sec`
- **Evaluation Time**: `7.28 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
