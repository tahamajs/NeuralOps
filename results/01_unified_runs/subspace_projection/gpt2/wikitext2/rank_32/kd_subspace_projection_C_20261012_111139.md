# Benchmark Run Summary: subspace_projection

- **Run ID**: `kd_subspace_projection_C_20261012_111139`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T11:14:09.888084+00:00
- **Git Commit**: `078042a45f` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `subspace_projection` |
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
| **Peak VRAM** | 2677.69 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module subspace_projection --model gpt2 --dataset wikitext2 --rank 32
```
