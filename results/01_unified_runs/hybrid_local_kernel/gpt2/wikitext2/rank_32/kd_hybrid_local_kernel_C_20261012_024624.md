# Benchmark Run Summary: hybrid_local_kernel

- **Run ID**: `kd_hybrid_local_kernel_C_20261012_024624`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T02:55:30.354480+00:00
- **Git Commit**: `2509d6bba0` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `hybrid_local_kernel` |
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
| **Peak VRAM** | 2822.13 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module hybrid_local_kernel --model gpt2 --dataset wikitext2 --rank 32
```
