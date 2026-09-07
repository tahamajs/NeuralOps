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
- **Validation Accuracy**: `82.00%`
- **F1 Score (Binary)**: `0.8265`
- **F1 Macro**: `0.8197`
- **Precision**: `0.8205`
- **Recall**: `0.8195`
- **Validation Loss**: `0.4033`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9977x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `127095564`
- **Trainable Recovery Params**: `2655756`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `339.82 MB`
- **Throughput**: `11301.92 tokens/sec`
- **Evaluation Time**: `25.57 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
