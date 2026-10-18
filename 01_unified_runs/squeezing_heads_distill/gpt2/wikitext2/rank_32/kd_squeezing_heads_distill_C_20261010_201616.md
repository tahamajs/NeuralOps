# Benchmark Run Summary: squeezing_heads_distill

- **Run ID**: `kd_squeezing_heads_distill_C_20261010_201616`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-10T16:49:29.535084+00:00
- **Git Commit**: `6e09c09271` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `squeezing_heads_distill` |
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
| **Peak VRAM** | 3376.84 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module squeezing_heads_distill --model gpt2 --dataset wikitext2 --rank 32
```
