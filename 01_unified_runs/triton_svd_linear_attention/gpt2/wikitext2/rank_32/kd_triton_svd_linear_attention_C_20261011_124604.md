# Benchmark Run Summary: triton_svd_linear_attention

- **Run ID**: `kd_triton_svd_linear_attention_C_20261011_124604`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T09:17:16.260836+00:00
- **Git Commit**: `0c80895808` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `triton_svd_linear_attention` |
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
| **Peak VRAM** | 1092.89 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module triton_svd_linear_attention --model gpt2 --dataset wikitext2 --rank 32
```
