# Benchmark Run Summary: gated_delta_net

- **Run ID**: `kd_gated_delta_net_C_20261012_013153`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T01:35:56.973520+00:00
- **Git Commit**: `96f078c31f` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `gated_delta_net` |
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
| **Peak VRAM** | 3002.04 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module gated_delta_net --model gpt2 --dataset wikitext2 --rank 32
```
