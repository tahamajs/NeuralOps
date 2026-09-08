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
- **Validation Accuracy**: `81.42%`
- **F1 Score (Binary)**: `0.8204`
- **F1 Macro**: `0.814`
- **Precision**: `0.8146`
- **Recall**: `0.8139`
- **Validation Loss**: `0.4239`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9968x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `127206288`
- **Trainable Recovery Params**: `2655744`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `996.93 MB`
- **Throughput**: `9245.92 tokens/sec`
- **Evaluation Time**: `27.75 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
