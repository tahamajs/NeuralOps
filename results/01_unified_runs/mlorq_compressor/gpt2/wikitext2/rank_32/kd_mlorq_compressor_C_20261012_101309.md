# Benchmark Run Summary: mlorq_compressor

- **Run ID**: `kd_mlorq_compressor_C_20261012_101309`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T10:16:31.122353+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `mlorq_compressor` |
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
| **Peak VRAM** | 2798.96 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module mlorq_compressor --model gpt2 --dataset wikitext2 --rank 32
```
