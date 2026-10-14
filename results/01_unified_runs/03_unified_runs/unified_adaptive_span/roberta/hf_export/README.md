---
tags:
- svd-linear-attention
- model-compression
- roberta
- pytorch
datasets:
- sst2
metrics:
- accuracy
- f1
---

# SVD Linear Attention / Compression: ROBERTA (roberta)

Validated factorized/adapted checkpoint of **roberta** using **roberta** evaluated on **SST-2**.

## Empirical Performance Results
- **Validation Accuracy**: `94.27%`
- **F1 Score (Binary)**: `0.9441`
- **F1 Macro**: `0.9426`
- **Precision**: `0.9428`
- **Recall**: `0.9425`
- **Validation Loss**: `0.2343`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `125532050`
- **Trainable Recovery Params**: `884880`
- **Compressed / Injected Layers**: `12`
- **Peak VRAM**: `653.51 MB`
- **Throughput**: `10181.82 tokens/sec`
- **Evaluation Time**: `25.75 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
