# Benchmark Run Summary: kmeans_quantize

- **Run ID**: `kd_kmeans_quantize_C_20261012_094331`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T09:46:26.368395+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `kmeans_quantize` |
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
| **Peak VRAM** | 2836.28 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module kmeans_quantize --model gpt2 --dataset wikitext2 --rank 32
```
