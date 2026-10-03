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
- **Validation Accuracy**: `74.89%`
- **F1 Score (Binary)**: `0.7812`
- **F1 Macro**: `0.7432`
- **Precision**: `0.7669`
- **Recall**: `0.7464`
- **Validation Loss**: `0.5276`
- **Quality Gate Status**: `PASS` (Pass Threshold: >= 56.00%)

## Efficiency Profile & Resource Footprint
- **Compression Rate**: `1.2304x`
- **FLOPs Reduction**: `0.0%`
- **Total Parameters Before**: `126800640`
- **Total Parameters After**: `103053648`
- **Trainable Recovery Params**: `4564560`
- **Compressed / Injected Layers**: `72`
- **Peak VRAM**: `300.59 MB`
- **Throughput**: `5372.02 tokens/sec`
- **Evaluation Time**: `95.78 s`

## Reproduction & Usage
```python
import torch

# Load the factorized model weights
checkpoint = torch.load("weights.pt", map_location="cpu")
print("Loaded parameter tensors successfully.")
```

Part of the SVD Linear Attention and LLM Compression Benchmark Suite.
