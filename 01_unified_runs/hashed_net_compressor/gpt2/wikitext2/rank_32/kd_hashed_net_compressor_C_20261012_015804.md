# Benchmark Run Summary: hashed_net_compressor

- **Run ID**: `kd_hashed_net_compressor_C_20261012_015804`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T01:58:30.166022+00:00
- **Git Commit**: `63acaaad85` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `hashed_net_compressor` |
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
| **Peak VRAM** | 1033.31 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module hashed_net_compressor --model gpt2 --dataset wikitext2 --rank 32
```
