# Benchmark Run Summary: residual_svd_compressor

- **Run ID**: `kd_residual_svd_compressor_C_20261011_160355`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T12:36:44.179128+00:00
- **Git Commit**: `0f67bb39bf` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `residual_svd_compressor` |
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
| **Peak VRAM** | 2516.76 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module residual_svd_compressor --model gpt2 --dataset wikitext2 --rank 32
```
