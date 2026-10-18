# Benchmark Run Summary: tomoe_compressor

- **Run ID**: `kd_tomoe_compressor_C_20261012_112748`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T11:31:14.902784+00:00
- **Git Commit**: `078042a45f` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `tomoe_compressor` |
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
| **Peak VRAM** | 3134.32 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module tomoe_compressor --model gpt2 --dataset wikitext2 --rank 32
```
