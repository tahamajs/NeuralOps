# Benchmark Run Summary: tri_branch_hybrid

- **Run ID**: `kd_tri_branch_hybrid_C_20261012_194340`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T19:44:07.812137+00:00
- **Git Commit**: `0d365d9f33` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `tri_branch_hybrid` |
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
| **Peak VRAM** | 1608.23 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module tri_branch_hybrid --model gpt2 --dataset wikitext2 --rank 32
```
