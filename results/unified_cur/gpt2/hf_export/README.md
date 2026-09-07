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
- **Validation Accuracy**: `83.03%`
- **F1 Score (Binary)**: `0.8442`
- **F1 Macro**: `0.8289`
- **Precision**: `0.8375`
- **Recall**: `0.8289`
- **Validation Loss**: `0.4792`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0013x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `126630864`
- **Trainable Recovery Params**: `2191056`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `411.93 MB`
- **Throughput**: `13474.58 tokens/sec`
- **Evaluation Time**: `41.89 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
