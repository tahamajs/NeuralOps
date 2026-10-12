# Benchmark Run Summary: split_head_hybrid

- **Run ID**: `kd_split_head_hybrid_C_20261012_063625`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T06:44:26.847438+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `split_head_hybrid` |
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
| **Peak VRAM** | 2862.16 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module split_head_hybrid --model gpt2 --dataset wikitext2 --rank 32
```
