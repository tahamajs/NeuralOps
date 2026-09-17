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
- **Validation Accuracy**: `81.19%`
- **F1 Score (Binary)**: `0.8233`
- **F1 Macro**: `0.8111`
- **Precision**: `0.8147`
- **Recall**: `0.811`
- **Validation Loss**: `0.4032`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `132913970`
- **Trainable Recovery Params**: `1179696`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `664.43 MB`
- **Throughput**: `11738.08 tokens/sec`
- **Evaluation Time**: `25.64 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
