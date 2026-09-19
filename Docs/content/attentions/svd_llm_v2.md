# Svd Llm V2

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
| **distilbert-sst2** | sst2 | `PASS` | 91.06% | **70.99%** | -20.07% | 0.7647 | 1.20x | 6 | [svd-linear-attention-distilbert-sst2-svd-llm-v2](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-svd-llm-v2) | 2026-09-19 |
| **distilbert-mnli-full** | mnli | `PASS` | 81.80% | **45.80%** | -36.00% | 0.4315 | 1.20x | 0 | N/A | 2026-09-19 |
| **distilbert-agnews-full** | ag_news | `PASS` | 94.30% | **86.90%** | -7.40% | 0.8662 | 1.20x | 0 | [svd-linear-attention-distilbert-agnews-full-svd-llm-v2](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-agnews-full-svd-llm-v2) | 2026-09-19 |