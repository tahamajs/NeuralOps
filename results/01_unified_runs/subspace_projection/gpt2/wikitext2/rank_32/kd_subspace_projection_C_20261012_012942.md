# Benchmark Run Summary: subspace_projection

- **Run ID**: `kd_subspace_projection_C_20261012_012942`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T01:29:58.453428+00:00
- **Git Commit**: `1143ec379d` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `subspace_projection` |
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
| **Peak VRAM** | 582.41 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module subspace_projection --model gpt2 --dataset wikitext2 --rank 32
```
