# Benchmark Run Summary: roast_hashed_linear

- **Run ID**: `kd_roast_hashed_linear_C_20261011_160710`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T12:39:56.583486+00:00
- **Git Commit**: `0f67bb39bf` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `roast_hashed_linear` |
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
| **Peak VRAM** | 2518.6 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module roast_hashed_linear --model gpt2 --dataset wikitext2 --rank 32
```
