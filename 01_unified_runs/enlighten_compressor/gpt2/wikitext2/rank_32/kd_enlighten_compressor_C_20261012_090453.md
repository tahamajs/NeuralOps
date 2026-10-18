# Benchmark Run Summary: enlighten_compressor

- **Run ID**: `kd_enlighten_compressor_C_20261012_090453`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T09:07:30.699207+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `enlighten_compressor` |
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
| **Peak VRAM** | 3095.31 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module enlighten_compressor --model gpt2 --dataset wikitext2 --rank 32
```
