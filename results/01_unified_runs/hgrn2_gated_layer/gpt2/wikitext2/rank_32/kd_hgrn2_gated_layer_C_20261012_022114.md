# Benchmark Run Summary: hgrn2_gated_layer

- **Run ID**: `kd_hgrn2_gated_layer_C_20261012_022114`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T02:30:00.978218+00:00
- **Git Commit**: `d9f7448602` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `hgrn2_gated_layer` |
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
| **Peak VRAM** | 3127.15 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module hgrn2_gated_layer --model gpt2 --dataset wikitext2 --rank 32
```
