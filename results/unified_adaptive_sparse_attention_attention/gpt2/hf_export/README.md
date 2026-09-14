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
- **Validation Accuracy**: `87.84%`
- **F1 Score (Binary)**: `0.8776`
- **F1 Macro**: `0.8784`
- **Precision**: `0.8791`
- **Recall**: `0.8789`
- **Validation Loss**: `0.3093`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.714x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `177590172`
- **Trainable Recovery Params**: `3539100`
- **Compressed / Injected Layers**: `96`
- **Peak VRAM**: `451.07 MB`
- **Throughput**: `7556.11 tokens/sec`
- **Evaluation Time**: `37.09 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
