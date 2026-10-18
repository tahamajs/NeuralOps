# Benchmark Run Summary: goldfinch_ahn

- **Run ID**: `kd_goldfinch_ahn_C_20261011_233154`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T23:35:55.572231+00:00
- **Git Commit**: `5ea385060b` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `goldfinch_ahn` |
| **Model** | `gpt2` |
| **Dataset** | `wikitext2` |
| **Rank** | `32` |
| **Epochs** | `1` |
| **Batch Size** | `12` |

## Primary Metrics
| Metric | Value |
|---|---|
| **Accuracy** | N/A |
| **F1 Score** | N/A |
| **Loss** | N/A |
| **Duration (s)** | 0.0s |
| **Peak VRAM** | 4680.98 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module goldfinch_ahn --model gpt2 --dataset wikitext2 --rank 32
```
