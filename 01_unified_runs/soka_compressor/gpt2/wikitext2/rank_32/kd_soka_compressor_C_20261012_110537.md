# Benchmark Run Summary: soka_compressor

- **Run ID**: `kd_soka_compressor_C_20261012_110537`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T11:08:17.173908+00:00
- **Git Commit**: `078042a45f` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `soka_compressor` |
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
| **Peak VRAM** | 2710.99 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module soka_compressor --model gpt2 --dataset wikitext2 --rank 32
```
