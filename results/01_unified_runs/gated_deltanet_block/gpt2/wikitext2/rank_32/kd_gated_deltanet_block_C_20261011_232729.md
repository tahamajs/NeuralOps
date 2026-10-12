# Benchmark Run Summary: gated_deltanet_block

- **Run ID**: `kd_gated_deltanet_block_C_20261011_232729`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T23:31:48.008110+00:00
- **Git Commit**: `75f4064a84` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `gated_deltanet_block` |
| **Model** | `gpt2` |
| **Dataset** | `wikitext2` |
| **Rank** | `32` |
| **Epochs** | `1` |
| **Batch Size** | `12` |

## Primary Metrics
| Metric | Value |
|---|---|
| **Accuracy** | N/A |
| **F1 Score** | N/A |
| **Loss** | N/A |
| **Duration (s)** | 0.0s |
| **Peak VRAM** | 4775.92 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module gated_deltanet_block --model gpt2 --dataset wikitext2 --rank 32
```
