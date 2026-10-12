# Benchmark Run Summary: soka_compressor

- **Run ID**: `kd_soka_compressor_C_20261012_013030`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T01:30:45.838827+00:00
- **Git Commit**: `1143ec379d` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `soka_compressor` |
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
| **Peak VRAM** | 575.19 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module soka_compressor --model gpt2 --dataset wikitext2 --rank 32
```
