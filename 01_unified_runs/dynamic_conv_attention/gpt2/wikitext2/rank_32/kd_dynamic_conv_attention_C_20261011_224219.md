# Benchmark Run Summary: dynamic_conv_attention

- **Run ID**: `kd_dynamic_conv_attention_C_20261011_224219`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T22:47:41.273179+00:00
- **Git Commit**: `55862cba9b` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `dynamic_conv_attention` |
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
| **Peak VRAM** | 4605.16 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module dynamic_conv_attention --model gpt2 --dataset wikitext2 --rank 32
```
