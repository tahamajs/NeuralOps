# Rpc

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
| **distilbert-sst2** | sst2 | `PASS` | 91.06% | **89.45%** | -1.61% | 0.8964 | 1.00x | 6 | [svd-linear-attention-distilbert-sst2-rpc](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-rpc) | 2026-09-19 |
| **distilbert-mnli-full** | mnli | `PASS` | 81.80% | **75.90%** | -5.90% | 0.7567 | 1.00x | 0 | [svd-linear-attention-distilbert-mnli-full-rpc](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-mnli-full-rpc) | 2026-09-19 |