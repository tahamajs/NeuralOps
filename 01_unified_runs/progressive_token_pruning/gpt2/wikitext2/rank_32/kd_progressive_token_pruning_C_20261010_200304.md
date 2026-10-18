# Benchmark Run Summary: progressive_token_pruning

- **Run ID**: `kd_progressive_token_pruning_C_20261010_200304`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-10T16:36:27.463995+00:00
- **Git Commit**: `6e09c09271` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `progressive_token_pruning` |
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
| **Peak VRAM** | 3379.23 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module progressive_token_pruning --model gpt2 --dataset wikitext2 --rank 32
```
