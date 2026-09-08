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
- **Validation Accuracy**: `80.73%`
- **F1 Score (Binary)**: `0.8178`
- **F1 Macro**: `0.8067`
- **Precision**: `0.8093`
- **Recall**: `0.8066`
- **Validation Loss**: `0.429`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.714x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `177590160`
- **Trainable Recovery Params**: `3540624`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `445.69 MB`
- **Throughput**: `44873.72 tokens/sec`
- **Evaluation Time**: `6.98 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
