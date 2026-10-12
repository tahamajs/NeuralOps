# Benchmark Run Summary: tri_branch_hybrid

- **Run ID**: `kd_tri_branch_hybrid_C_20261012_211524`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T21:19:20.676504+00:00
- **Git Commit**: `d4e42905d3` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `tri_branch_hybrid` |
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
| **Peak VRAM** | 3397.63 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module tri_branch_hybrid --model gpt2 --dataset wikitext2 --rank 32
```
