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
- **F1 Score (Binary)**: `0.8455`
- **F1 Macro**: `0.8286`
- **Precision**: `0.8395`
- **Recall**: `0.8287`
- **Validation Loss**: `0.4118`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9432x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `134440848`
- **Trainable Recovery Params**: `2950800`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `358.28 MB`
- **Throughput**: `34167.48 tokens/sec`
- **Evaluation Time**: `8.84 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
