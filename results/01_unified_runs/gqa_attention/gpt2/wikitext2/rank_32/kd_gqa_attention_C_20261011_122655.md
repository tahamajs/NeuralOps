# Benchmark Run Summary: gqa_attention

- **Run ID**: `kd_gqa_attention_C_20261011_122655`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T08:57:58.088533+00:00
- **Git Commit**: `3476950323` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `gqa_attention` |
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
| **Peak VRAM** | 1108.59 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module gqa_attention --model gpt2 --dataset wikitext2 --rank 32
```
