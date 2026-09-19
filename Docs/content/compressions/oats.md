# Oats

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
| **distilbert-sst2** | sst2 | `PASS` | 91.06% | **90.60%** | -0.46% | 0.9074 | 1.00x | 36 | [svd-linear-attention-distilbert-sst2-oats](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-oats) | 2026-09-19 |
| **distilbert-mnli-full** | mnli | `PASS` | 81.80% | **81.60%** | -0.20% | 0.8154 | 1.00x | 36 | N/A | 2026-09-19 |