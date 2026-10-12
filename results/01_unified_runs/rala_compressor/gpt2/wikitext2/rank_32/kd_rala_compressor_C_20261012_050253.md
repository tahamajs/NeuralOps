# Benchmark Run Summary: rala_compressor

- **Run ID**: `kd_rala_compressor_C_20261012_050253`
- **Status**: **FAILED**
- **Timestamp**: 2026-10-12T05:11:32.892564+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `rala_compressor` |
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
| **Peak VRAM** | 2843.0 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module rala_compressor --model gpt2 --dataset wikitext2 --rank 32
```
