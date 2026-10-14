# Benchmark Run Summary: proxsparse

- **Run ID**: `41bd6f4a-e6cb-4d87-a42a-5b5d7bd0bbd6`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-14T05:44:11Z
- **Git Commit**: `32e519631c` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `proxsparse` |
| **Model** | `gpt2` |
| **Dataset** | `sst2` |
| **Rank** | `64` |
| **Training Type** | `finetune` |
| **Batch Size** | `2` |

## Primary Metrics
| Metric | Value |
|---|---|
| **Pre-KD Perplexity** | N/A |
| **Post-KD Perplexity** | N/A |
| **Recovery %** | N/A |
| **KD Gain** | N/A |
| **Accuracy** | 0.3500 |
| **F1 Score** | 0.2353 |
| **Loss** | 1.1253 |
| **Duration (s)** | 12.89s |
| **Peak VRAM** | 994.38525390625 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module proxsparse --model gpt2 --dataset sst2 --rank 64
```
