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
- **Validation Accuracy**: `58.26%`
- **F1 Score (Binary)**: `0.6818`
- **F1 Macro**: `0.5376`
- **Precision**: `0.6216`
- **Recall**: `0.577`
- **Validation Loss**: `0.6713`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9214x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `135273230`
- **Trainable Recovery Params**: `3538956`
- **Compressed / Injected Layers**: `48`
- **Peak VRAM**: `681.6 MB`
- **Throughput**: `12047.43 tokens/sec`
- **Evaluation Time**: `62.36 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
