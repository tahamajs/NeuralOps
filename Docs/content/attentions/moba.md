# Moba

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
| **distilbert-sst2** | sst2 | `PASS` | 91.06% | **90.71%** | -0.35% | 0.9105 | 1.00x | 6 | [svd-linear-attention-distilbert-sst2-moba](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-moba) | 2026-09-19 |
