# Benchmark Run Summary: lsc_compressor

- **Run ID**: `kd_lsc_compressor_C_20261011_231538`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T23:15:50.483744+00:00
- **Git Commit**: `d7633f0b1b` (dirty=True)

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
| **Peak VRAM** | 1354.41 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module lsc_compressor --model gpt2 --dataset wikitext2 --rank 32
```
