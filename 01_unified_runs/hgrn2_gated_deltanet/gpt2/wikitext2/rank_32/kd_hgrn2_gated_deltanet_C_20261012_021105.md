# Benchmark Run Summary: hgrn2_gated_deltanet

- **Run ID**: `kd_hgrn2_gated_deltanet_C_20261012_021105`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T02:20:46.116010+00:00
- **Git Commit**: `d9f7448602` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `hgrn2_gated_deltanet` |
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
| **Peak VRAM** | 3324.79 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module hgrn2_gated_deltanet --model gpt2 --dataset wikitext2 --rank 32
```
