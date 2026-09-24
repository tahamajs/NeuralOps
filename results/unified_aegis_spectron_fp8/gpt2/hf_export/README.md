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
- **Validation Accuracy**: `72.36%`
- **F1 Score (Binary)**: `0.7771`
- **F1 Macro**: `0.7068`
- **Precision**: `0.7786`
- **Recall**: `0.7195`
- **Validation Loss**: `0.5351`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1278x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `110341044`
- **Trainable Recovery Params**: `2368512`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `1133.95 MB`
- **Throughput**: `90.96 tokens/sec`
- **Evaluation Time**: `1537.66 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
