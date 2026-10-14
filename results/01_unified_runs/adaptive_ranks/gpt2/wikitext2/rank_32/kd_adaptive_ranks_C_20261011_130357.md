# Benchmark Run Summary: adaptive_ranks

- **Run ID**: `kd_adaptive_ranks_C_20261011_130357`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T09:36:49.980092+00:00
- **Git Commit**: `a2276b9f36` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `adaptive_ranks` |
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
| **Peak VRAM** | 2516.83 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module adaptive_ranks --model gpt2 --dataset wikitext2 --rank 32
```
