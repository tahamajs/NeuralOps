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
- **Validation Accuracy**: `82.57%`
- **F1 Score (Binary)**: `0.8362`
- **F1 Macro**: `0.825`
- **Precision**: `0.8287`
- **Recall**: `0.8248`
- **Validation Loss**: `0.4418`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9373x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `132990806`
- **Trainable Recovery Params**: `1820978`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `387.78 MB`
- **Throughput**: `17723.62 tokens/sec`
- **Evaluation Time**: `16.33 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
