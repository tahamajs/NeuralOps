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
- **Validation Accuracy**: `75.00%`
- **F1 Score (Binary)**: `0.7578`
- **F1 Macro**: `0.7497`
- **Precision**: `0.7501`
- **Recall**: `0.7497`
- **Validation Loss**: `0.5699`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.0x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124647170`
- **Total Parameters After**: `124647170`
- **Trainable Recovery Params**: `0`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `712.34 MB`
- **Throughput**: `28181.43 tokens/sec`
- **Evaluation Time**: `6.01 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
