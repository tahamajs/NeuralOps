# Benchmark Run Summary: oats_compressor

- **Run ID**: `kd_oats_compressor_C_20261011_155401`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T12:27:03.294204+00:00
- **Git Commit**: `0f67bb39bf` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `oats_compressor` |
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
| **Peak VRAM** | 2515.7 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module oats_compressor --model gpt2 --dataset wikitext2 --rank 32
```
