# Sdrm

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

## Empirical Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distilbert-agnews-full** | ag_news | `PASS` | 94.30% | **91.50%** | -2.80% | 0.9117 | 1.21x | 0 | [svd-linear-attention-distilbert-agnews-full-sdrm](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-agnews-full-sdrm) | 2026-09-19 |
