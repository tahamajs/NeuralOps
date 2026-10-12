# Benchmark Run Summary: tri_branch_hybrid

- **Run ID**: `kd_tri_branch_hybrid_C_20261012_195121`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T19:52:18.154953+00:00
- **Git Commit**: `4904ad5364` (dirty=True)

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
| **Peak VRAM** | 1606.73 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module tri_branch_hybrid --model gpt2 --dataset wikitext2 --rank 32
```
