# Benchmark Run Summary: fused_svd_kernel

- **Run ID**: `kd_fused_svd_kernel_C_20261011_123628`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T09:10:38.483709+00:00
- **Git Commit**: `3476950323` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `fused_svd_kernel` |
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
| **Peak VRAM** | 1061.64 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module fused_svd_kernel --model gpt2 --dataset wikitext2 --rank 32
```
