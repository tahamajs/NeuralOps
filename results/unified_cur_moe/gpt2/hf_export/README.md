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
- **Validation Accuracy**: `58.94%`
- **F1 Score (Binary)**: `0.6961`
- **F1 Macro**: `0.5318`
- **Precision**: `0.6561`
- **Recall**: `0.5832`
- **Validation Loss**: `0.6754`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.401x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `90508800`
- **Trainable Recovery Params**: `2008320`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `343.64 MB`
- **Throughput**: `10875.25 tokens/sec`
- **Evaluation Time**: `33.89 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
