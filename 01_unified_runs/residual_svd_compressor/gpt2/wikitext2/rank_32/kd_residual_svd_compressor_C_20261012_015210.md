# Benchmark Run Summary: residual_svd_compressor

- **Run ID**: `kd_residual_svd_compressor_C_20261012_015210`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T01:52:32.515828+00:00
- **Git Commit**: `b81e4284d3` (dirty=True)

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
| **Peak VRAM** | 506.9 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module residual_svd_compressor --model gpt2 --dataset wikitext2 --rank 32
```
