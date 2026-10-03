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
- **Validation Accuracy**: `81.88%`
- **F1 Score (Binary)**: `0.8326`
- **F1 Macro**: `0.8176`
- **Precision**: `0.8245`
- **Recall**: `0.8176`
- **Validation Loss**: `0.381`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `133614530`
- **Trainable Recovery Params**: `1290432`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `669.6 MB`
- **Throughput**: `8397.67 tokens/sec`
- **Evaluation Time**: `34.52 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
