# Benchmark Run Summary: gated_hybrid_kernel

- **Run ID**: `kd_gated_hybrid_kernel_C_20261010_053735`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-10T02:22:32.219969+00:00
- **Git Commit**: `3e5c8ecdb3` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `gated_hybrid_kernel` |
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
| **Peak VRAM** | 2964.97 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module gated_hybrid_kernel --model gpt2 --dataset wikitext2 --rank 32
```
