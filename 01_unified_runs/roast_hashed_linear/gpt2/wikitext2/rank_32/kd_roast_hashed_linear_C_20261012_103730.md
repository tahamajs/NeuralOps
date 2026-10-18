# Benchmark Run Summary: roast_hashed_linear

- **Run ID**: `kd_roast_hashed_linear_C_20261012_103730`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T10:39:13.755808+00:00
- **Git Commit**: `476b4c8262` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `roast_hashed_linear` |
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
| **Peak VRAM** | 2827.01 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module roast_hashed_linear --model gpt2 --dataset wikitext2 --rank 32
```
