# Benchmark Run Summary: dynamic_conv_attention

- **Run ID**: `kd_dynamic_conv_attention_C_20261010_023805`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-09T23:13:19.573148+00:00
- **Git Commit**: `53935d1ead` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `dynamic_conv_attention` |
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
| **Peak VRAM** | 3382.08 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module dynamic_conv_attention --model gpt2 --dataset wikitext2 --rank 32
```
