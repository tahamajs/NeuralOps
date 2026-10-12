# Benchmark Run Summary: gated_linear_attention

- **Run ID**: `kd_gated_linear_attention_C_20261012_193525`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T19:36:06.954131+00:00
- **Git Commit**: `0d365d9f33` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `gated_linear_attention` |
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
| **Peak VRAM** | 2842.67 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module gated_linear_attention --model gpt2 --dataset wikitext2 --rank 32
```
