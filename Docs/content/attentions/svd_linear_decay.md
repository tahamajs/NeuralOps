# Svd Linear Decay

## Overview
Decay Gated SVD Linear Attention.

## Configuration & Hyperparameters

| Parameter | Description |
|-----------|-------------|

## Usage Examples

```python
from Codes.framework.attentions import DecayGatedLinearAttention
```

## Benchmark Results

| Model | Dataset | Status | Baseline Acc | Injected Acc | Latency | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| BERT | sst2 | `PASS` | 91.06% | **0.00%** | 14.24 ms | 719.1 MB |
| BERT | mrpc | `PASS` | 91.06% | **0.00%** | 14.89 ms | 719.1 MB |
| ROBERTA | sst2 | `PASS` | 91.06% | **0.00%** | 15.73 ms | 729.0 MB |
| ROBERTA | mrpc | `PASS` | 91.06% | **0.00%** | 15.01 ms | 729.0 MB |

## Limitations & Future Work

TODO: Identify specific limitations and edge cases for this module.

## Empirical Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distilbert-sst2** | sst2 | `PASS` | 91.06% | **84.52%** | -6.54% | 0.8525 | 1.20x | 6 | [svd-linear-attention-distilbert-sst2-svd-linear-decay](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-svd-linear-decay) | 2026-09-19 |
| **distilbert-mnli-full** | mnli | `PASS` | 81.80% | **53.90%** | -27.90% | 0.5017 | 1.20x | 6 | N/A | 2026-09-19 |
| **distilbert-agnews-full** | ag_news | `PASS` | 94.30% | **89.40%** | -4.90% | 0.8903 | 1.20x | 6 | [svd-linear-attention-distilbert-agnews-full-svd-linear-decay](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-agnews-full-svd-linear-decay) | 2026-09-19 |