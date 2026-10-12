# Benchmark Run Summary: hybrid_svd_layer

- **Run ID**: `kd_hybrid_svd_layer_C_20261012_094044`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T09:43:17.323103+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `hybrid_svd_layer` |
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
| **Peak VRAM** | 2762.69 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module hybrid_svd_layer --model gpt2 --dataset wikitext2 --rank 32
```
