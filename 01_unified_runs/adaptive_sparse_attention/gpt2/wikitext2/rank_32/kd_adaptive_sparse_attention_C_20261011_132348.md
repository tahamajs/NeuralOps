# Benchmark Run Summary: adaptive_sparse_attention

- **Run ID**: `kd_adaptive_sparse_attention_C_20261011_132348`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T09:55:40.129256+00:00
- **Git Commit**: `363d9d95c7` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `adaptive_sparse_attention` |
| **Model** | `gpt2` |
| **Dataset** | `wikitext2` |
| **Rank** | `32` |
| **Epochs** | `1` |
| **Batch Size** | `4` |

## Primary Metrics
| Metric | Value |
|---|---|
| **Accuracy** | N/A |
| **F1 Score** | N/A |
| **Loss** | N/A |
| **Duration (s)** | 0.0s |
| **Peak VRAM** | 1146.94 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module adaptive_sparse_attention --model gpt2 --dataset wikitext2 --rank 32
```
