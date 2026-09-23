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
- **Validation Accuracy**: `70.76%`
- **F1 Score (Binary)**: `0.7615`
- **F1 Macro**: `0.6918`
- **Precision**: `0.7507`
- **Recall**: `0.7037`
- **Validation Loss**: `0.5508`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9931x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `127685568`
- **Trainable Recovery Params**: `3244224`
- **Compressed / Injected Layers**: `72`
- **Peak VRAM**: `349.49 MB`
- **Throughput**: `8689.53 tokens/sec`
- **Evaluation Time**: `51.12 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
