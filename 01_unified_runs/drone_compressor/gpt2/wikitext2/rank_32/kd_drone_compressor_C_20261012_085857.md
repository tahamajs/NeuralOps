# Benchmark Run Summary: drone_compressor

- **Run ID**: `kd_drone_compressor_C_20261012_085857`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T09:02:08.796224+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `drone_compressor` |
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
| **Peak VRAM** | 2782.99 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module drone_compressor --model gpt2 --dataset wikitext2 --rank 32
```
