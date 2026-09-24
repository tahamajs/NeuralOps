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
- **Validation Accuracy**: `70.30%`
- **F1 Score (Binary)**: `0.7277`
- **F1 Macro**: `0.7005`
- **Precision**: `0.707`
- **Recall**: `0.7016`
- **Validation Loss**: `0.5672`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `125620992`
- **Trainable Recovery Params**: `1179648`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `433.51 MB`
- **Throughput**: `1913.43 tokens/sec`
- **Evaluation Time**: `161.36 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
