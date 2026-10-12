# Benchmark Run Summary: lillama_compressor

- **Run ID**: `kd_lillama_compressor_C_20261012_100016`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T10:03:34.726917+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `lillama_compressor` |
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
| **Peak VRAM** | 3095.75 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module lillama_compressor --model gpt2 --dataset wikitext2 --rank 32
```
