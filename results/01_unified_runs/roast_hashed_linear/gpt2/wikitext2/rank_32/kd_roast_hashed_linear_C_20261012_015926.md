# Benchmark Run Summary: roast_hashed_linear

- **Run ID**: `kd_roast_hashed_linear_C_20261012_015926`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T01:59:44.516188+00:00
- **Git Commit**: `76c618b17a` (dirty=True)

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
| **Peak VRAM** | 511.85 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module roast_hashed_linear --model gpt2 --dataset wikitext2 --rank 32
```
