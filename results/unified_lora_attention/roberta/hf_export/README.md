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
- **Validation Accuracy**: `84.63%`
- **F1 Score (Binary)**: `0.8649`
- **F1 Macro**: `0.8434`
- **Precision**: `0.8683`
- **Recall**: `0.8441`
- **Validation Loss**: `0.4875`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.937x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `127325954`
- **Total Parameters After**: `135887618`
- **Trainable Recovery Params**: `4745474`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `411.84 MB`
- **Throughput**: `10456.4 tokens/sec`
- **Evaluation Time**: `29.2 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
