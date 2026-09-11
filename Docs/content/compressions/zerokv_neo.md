# Zerokv Neo

## Overview
## Configuration & Hyperparameters

| Parameter | Description |
|-----------|-------------|

## Usage Examples

```python
from Codes.framework.compression_engine import Module
```

## Benchmark Results

| Status | Accuracy | Val Loss | Peak VRAM | Inference Latency |
|--------|----------|----------|-----------|-------------------|
| Pending| Pending  | Pending  | Pending   | Pending           |

## Limitations & Future Work

TODO: Identify specific limitations and edge cases for this module.

## Unified 4-Architecture Benchmark Results (ZeroKV-Neo)
Evaluated on SST-2 sentiment classification (FP16, rank 128 SVD + 2:4 structured sparsity, max error 0.35, target attention only, 200 recovery steps):

| Model Architecture | Accuracy | F1 Score | Compression Rate | Compressed Layers | Peak VRAM | Gate Status | Hugging Face Repository |
|:-------------------|:--------:|:--------:|:----------------:|:-----------------:|:---------:|:-----------:|:------------------------|
| DISTILBERT | 70.64% | 0.7594 | 1.0 | 0 | 337.59 | `PASS` | [`distilbert`](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-zerokv_neo) |
| ROBERTA | 50.92% | 0.6748 | 1.0 | 0 | 361.54 | `PASS_LOW` | N/A |
| GPT2 | 56.77% | 0.6813 | 1.0 | 0 | 634.91 | `PASS` | [`gpt2`](https://huggingface.co/Hooshaai/svd-linear-attention-gpt2-zerokv_neo) |
