# Benchmark Run Summary: lsc_compressor

- **Run ID**: `kd_lsc_compressor_C_20261012_014506`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T01:45:20.417994+00:00
- **Git Commit**: `89cbe53832` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `lsc_compressor` |
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
| **Peak VRAM** | 879.71 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module lsc_compressor --model gpt2 --dataset wikitext2 --rank 32
```
