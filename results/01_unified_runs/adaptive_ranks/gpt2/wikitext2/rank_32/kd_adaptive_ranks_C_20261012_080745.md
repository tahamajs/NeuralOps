# Benchmark Run Summary: adaptive_ranks

- **Run ID**: `kd_adaptive_ranks_C_20261012_080745`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T08:10:45.543399+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `adaptive_ranks` |
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
| **Peak VRAM** | 3092.04 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module adaptive_ranks --model gpt2 --dataset wikitext2 --rank 32
```
