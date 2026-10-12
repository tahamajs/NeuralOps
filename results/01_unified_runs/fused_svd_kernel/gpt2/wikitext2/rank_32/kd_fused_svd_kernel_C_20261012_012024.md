# Benchmark Run Summary: fused_svd_kernel

- **Run ID**: `kd_fused_svd_kernel_C_20261012_012024`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T01:24:30.174840+00:00
- **Git Commit**: `47f3447b65` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `fused_svd_kernel` |
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
| **Peak VRAM** | 2824.84 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module fused_svd_kernel --model gpt2 --dataset wikitext2 --rank 32
```
