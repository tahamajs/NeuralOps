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
| DISTILBERT | 75.92% | 0.7925 | 1.0 | 0 | 312.19 | `PASS` | [`distilbert`](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-zerokv_neo) |
| ROBERTA | 50.92% | 0.6748 | 1.0 | 0 | 531.69 | `PASS_LOW` | N/A |
| GPT2 | 51.15% | 0.6758 | 1.0 | 48 | 413.5 | `PASS_LOW` | N/A |

## Empirical Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distilbert-sst2** | sst2 | `PASS` | 91.06% | **79.36%** | -11.70% | 0.7862 | 1.00x | 0 | [svd-linear-attention-distilbert-sst2-zerokv](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-zerokv) | 2026-09-19 |
