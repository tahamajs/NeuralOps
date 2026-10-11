# Benchmark Run Summary: optimal_subspace_projection

- **Run ID**: `kd_optimal_subspace_projection_C_20261011_155731`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T12:30:19.909786+00:00
- **Git Commit**: `0f67bb39bf` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `optimal_subspace_projection` |
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
| **Peak VRAM** | 2517.31 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module optimal_subspace_projection --model gpt2 --dataset wikitext2 --rank 32
```
