# Benchmark Run Summary: bonsai_pruner

- **Run ID**: `kd_bonsai_pruner_C_20261012_082831`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T08:31:25.650900+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `bonsai_pruner` |
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
| **Peak VRAM** | 3258.12 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module bonsai_pruner --model gpt2 --dataset wikitext2 --rank 32
```
