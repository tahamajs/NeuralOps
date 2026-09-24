# Mamba Inject

## Overview
## Configuration & Hyperparameters

| Parameter | Description |
|-----------|-------------|

## Usage Examples

```python
from Codes.framework.attentions import Module
```

## Benchmark Results

| Model | Dataset | Status | Baseline Acc | Injected Acc | Latency | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| BERT | sst2 | `PASS` | 91.06% | **0.00%** | 8.15 ms | 466.6 MB |
| BERT | mrpc | `PASS` | 91.06% | **0.00%** | 7.44 ms | 466.6 MB |
| ROBERTA | sst2 | `PASS` | 91.06% | **0.00%** | 8.19 ms | 476.5 MB |
| ROBERTA | mrpc | `PASS` | 91.06% | **0.00%** | 8.18 ms | 476.5 MB |

## Limitations & Future Work

TODO: Identify specific limitations and edge cases for this module.
