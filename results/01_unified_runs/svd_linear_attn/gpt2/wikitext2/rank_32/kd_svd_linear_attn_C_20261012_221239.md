# Benchmark Run Summary: svd_linear_attn

- **Run ID**: `kd_svd_linear_attn_C_20261012_221239`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T22:14:53.823000+00:00
- **Git Commit**: `f0f7dc2225` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `svd_linear_attn` |
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
| **Peak VRAM** | 5382.41 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module svd_linear_attn --model gpt2 --dataset wikitext2 --rank 32
```
