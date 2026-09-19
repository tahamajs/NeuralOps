# Cur Moe

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
| **distilbert-sst2** | sst2 | `PASS` | 91.06% | **87.73%** | -3.33% | 0.8807 | 1.06x | 0 | [svd-linear-attention-distilbert-sst2-cur-moe](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-cur-moe) | 2026-09-19 |
| **distilbert-mnli-full** | mnli | `PASS` | 81.80% | **75.90%** | -5.90% | 0.7575 | 1.06x | 0 | N/A | 2026-09-19 |