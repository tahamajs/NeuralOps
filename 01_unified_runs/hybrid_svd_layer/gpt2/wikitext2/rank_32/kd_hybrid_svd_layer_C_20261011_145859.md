# Benchmark Run Summary: hybrid_svd_layer

- **Run ID**: `kd_hybrid_svd_layer_C_20261011_145859`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T11:32:19.083709+00:00
- **Git Commit**: `0f67bb39bf` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `hybrid_svd_layer` |
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
| **Peak VRAM** | 2148.7 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module hybrid_svd_layer --model gpt2 --dataset wikitext2 --rank 32
```
