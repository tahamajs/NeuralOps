# Benchmark Run Summary: bonsai_pruner

- **Run ID**: `kd_bonsai_pruner_C_20261011_132725`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T10:00:33.885115+00:00
- **Git Commit**: `e0119bde06` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `bonsai_pruner` |
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
| **Peak VRAM** | 2633.95 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module bonsai_pruner --model gpt2 --dataset wikitext2 --rank 32
```
