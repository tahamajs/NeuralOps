# Benchmark Run Summary: svd_linear_decay

- **Run ID**: `kd_svd_linear_decay_C_20261012_065051`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T06:53:23.306233+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `svd_linear_decay` |
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
| **Peak VRAM** | 2779.81 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module svd_linear_decay --model gpt2 --dataset wikitext2 --rank 32
```
