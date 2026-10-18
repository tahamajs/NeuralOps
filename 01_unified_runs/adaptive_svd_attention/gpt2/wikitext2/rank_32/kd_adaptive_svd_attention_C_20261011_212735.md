# Benchmark Run Summary: adaptive_svd_attention

- **Run ID**: `kd_adaptive_svd_attention_C_20261011_212735`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T21:30:10.477824+00:00
- **Git Commit**: `acb14d14b1` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `adaptive_svd_attention` |
| **Model** | `gpt2` |
| **Dataset** | `wikitext2` |
| **Rank** | `32` |
| **Epochs** | `1` |
| **Batch Size** | `8` |

## Primary Metrics
| Metric | Value |
|---|---|
| **Accuracy** | N/A |
| **F1 Score** | N/A |
| **Loss** | N/A |
| **Duration (s)** | 0.0s |
| **Peak VRAM** | 3340.96 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module adaptive_svd_attention --model gpt2 --dataset wikitext2 --rank 32
```
