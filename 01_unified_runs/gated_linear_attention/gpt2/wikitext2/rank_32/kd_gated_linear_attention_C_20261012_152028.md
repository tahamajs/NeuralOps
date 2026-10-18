# Benchmark Run Summary: gated_linear_attention

- **Run ID**: `kd_gated_linear_attention_C_20261012_152028`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T15:23:13.482321+00:00
- **Git Commit**: `6d9b481f76` (dirty=True)

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
| **Peak VRAM** | 2837.34 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module gated_linear_attention --model gpt2 --dataset wikitext2 --rank 32
```
