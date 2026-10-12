# Benchmark Run Summary: come_compressor

- **Run ID**: `kd_come_compressor_C_20261012_083436`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T08:37:18.312428+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `come_compressor` |
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
| **Peak VRAM** | 2780.27 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module come_compressor --model gpt2 --dataset wikitext2 --rank 32
```
