# Squeezing Heads

## Overview
## Configuration & Hyperparameters

| Parameter | Description |
|-----------|-------------|

## Usage Examples

```python
from Codes.framework.attentions import Module
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
| **distilbert-sst2** | sst2 | `PASS` | 91.06% | **90.71%** | -0.35% | 0.9091 | 1.00x | 6 | [svd-linear-attention-distilbert-sst2-squeezing-heads](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-squeezing-heads) | 2026-09-19 |
| **distilbert-agnews-full** | ag_news | `PASS` | 94.30% | **94.30%** | +0.00% | 0.9391 | 1.00x | 0 | N/A | 2026-09-19 |