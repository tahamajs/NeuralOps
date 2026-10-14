# Benchmark Run Summary: adasvd_compressor

- **Run ID**: `kd_adasvd_compressor_C_20261011_130719`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T09:40:16.182527+00:00
- **Git Commit**: `a2276b9f36` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `adasvd_compressor` |
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
| **Peak VRAM** | 2457.99 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module adasvd_compressor --model gpt2 --dataset wikitext2 --rank 32
```
