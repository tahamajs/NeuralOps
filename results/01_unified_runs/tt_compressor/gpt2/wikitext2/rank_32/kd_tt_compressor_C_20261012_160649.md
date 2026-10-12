# Benchmark Run Summary: tt_compressor

- **Run ID**: `kd_tt_compressor_C_20261012_160649`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T16:08:55.342594+00:00
- **Git Commit**: `4e602dd050` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `tt_compressor` |
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
| **Peak VRAM** | 3002.31 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module tt_compressor --model gpt2 --dataset wikitext2 --rank 32
```
