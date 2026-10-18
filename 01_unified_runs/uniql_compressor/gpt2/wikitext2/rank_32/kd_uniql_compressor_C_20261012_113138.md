# Benchmark Run Summary: uniql_compressor

- **Run ID**: `kd_uniql_compressor_C_20261012_113138`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T11:34:24.117914+00:00
- **Git Commit**: `078042a45f` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `uniql_compressor` |
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
| **Peak VRAM** | 2688.07 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module uniql_compressor --model gpt2 --dataset wikitext2 --rank 32
```
