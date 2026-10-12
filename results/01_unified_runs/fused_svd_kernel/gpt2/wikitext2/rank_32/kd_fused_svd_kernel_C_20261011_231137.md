# Benchmark Run Summary: fused_svd_kernel

- **Run ID**: `kd_fused_svd_kernel_C_20261011_231137`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T23:15:32.269706+00:00
- **Git Commit**: `d7633f0b1b` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `fused_svd_kernel` |
| **Model** | `gpt2` |
| **Dataset** | `wikitext2` |
| **Rank** | `32` |
| **Epochs** | `1` |
| **Batch Size** | `12` |

## Primary Metrics
| Metric | Value |
|---|---|
| **Accuracy** | N/A |
| **F1 Score** | N/A |
| **Loss** | N/A |
| **Duration (s)** | 0.0s |
| **Peak VRAM** | 4587.73 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module fused_svd_kernel --model gpt2 --dataset wikitext2 --rank 32
```
