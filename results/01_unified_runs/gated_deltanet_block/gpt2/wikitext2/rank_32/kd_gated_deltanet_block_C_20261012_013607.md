# Benchmark Run Summary: gated_deltanet_block

- **Run ID**: `kd_gated_deltanet_block_C_20261012_013607`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T01:40:00.225353+00:00
- **Git Commit**: `2991eb4bd6` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `gated_deltanet_block` |
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
| **Peak VRAM** | 2998.87 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module gated_deltanet_block --model gpt2 --dataset wikitext2 --rank 32
```
