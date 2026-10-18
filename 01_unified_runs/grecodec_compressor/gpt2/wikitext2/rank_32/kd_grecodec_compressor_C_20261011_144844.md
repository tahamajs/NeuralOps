# Benchmark Run Summary: grecodec_compressor

- **Run ID**: `kd_grecodec_compressor_C_20261011_144844`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T11:21:59.635361+00:00
- **Git Commit**: `0f67bb39bf` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `grecodec_compressor` |
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
| **Peak VRAM** | 2050.87 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module grecodec_compressor --model gpt2 --dataset wikitext2 --rank 32
```
