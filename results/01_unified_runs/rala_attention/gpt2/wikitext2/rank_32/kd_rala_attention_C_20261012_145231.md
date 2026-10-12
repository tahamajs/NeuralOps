# Benchmark Run Summary: rala_attention

- **Run ID**: `kd_rala_attention_C_20261012_145231`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T14:54:09.603503+00:00
- **Git Commit**: `38527665ca` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `rala_attention` |
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
| **Peak VRAM** | 2840.0 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module rala_attention --model gpt2 --dataset wikitext2 --rank 32
```
