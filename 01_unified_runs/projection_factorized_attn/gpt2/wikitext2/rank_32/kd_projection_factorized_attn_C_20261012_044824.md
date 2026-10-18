# Benchmark Run Summary: projection_factorized_attn

- **Run ID**: `kd_projection_factorized_attn_C_20261012_044824`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T04:50:59.179521+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `projection_factorized_attn` |
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
| **Peak VRAM** | 2780.86 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module projection_factorized_attn --model gpt2 --dataset wikitext2 --rank 32
```
