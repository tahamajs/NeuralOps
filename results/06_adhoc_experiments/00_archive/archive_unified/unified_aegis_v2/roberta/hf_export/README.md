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
- **Validation Accuracy**: `62.39%`
- **F1 Score (Binary)**: `0.7163`
- **F1 Macro**: `0.5792`
- **Precision**: `0.697`
- **Recall**: `0.6181`
- **Validation Loss**: `0.6316`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.1449x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `108872018`
- **Trainable Recovery Params**: `3090000`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `591.16 MB`
- **Throughput**: `7112.56 tokens/sec`
- **Evaluation Time**: `46.28 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
