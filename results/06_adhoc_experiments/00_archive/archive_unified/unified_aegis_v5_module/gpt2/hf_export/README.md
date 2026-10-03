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
- **Validation Accuracy**: `56.31%`
- **F1 Score (Binary)**: `0.6495`
- **F1 Macro**: `0.5348`
- **Precision**: `0.5758`
- **Recall**: `0.5587`
- **Validation Loss**: `0.7094`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `0.9875x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `124441344`
- **Total Parameters After**: `126014400`
- **Trainable Recovery Params**: `1574592`
- **Compressed / Injected Layers**: `0`
- **Peak VRAM**: `595.01 MB`
- **Throughput**: `44929.72 tokens/sec`
- **Evaluation Time**: `15.99 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
