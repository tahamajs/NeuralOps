# Benchmark Run Summary: progressive_token_pruning

- **Run ID**: `kd_progressive_token_pruning_C_20261012_044512`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T04:48:09.959371+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `progressive_token_pruning` |
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
| **Peak VRAM** | 2837.78 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module progressive_token_pruning --model gpt2 --dataset wikitext2 --rank 32
```
