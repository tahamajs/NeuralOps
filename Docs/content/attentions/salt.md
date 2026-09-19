# Salt

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
| **distilbert-sst2** | sst2 | `PASS` | 91.06% | **84.29%** | -6.77% | 0.8496 | 1.21x | 6 | [svd-linear-attention-distilbert-sst2-salt](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-salt) | 2026-09-19 |
| **distilbert-mnli-full** | mnli | `PASS` | 81.80% | **52.00%** | -29.80% | 0.4652 | 1.21x | 0 | N/A | 2026-09-19 |