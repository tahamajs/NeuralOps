# Benchmark Run Summary: adaptive_span_attention

- **Run ID**: `kd_adaptive_span_attention_C_20261011_124348`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T09:15:10.009715+00:00
- **Git Commit**: `0c80895808` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `adaptive_span_attention` |
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
| **Peak VRAM** | 1146.93 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module adaptive_span_attention --model gpt2 --dataset wikitext2 --rank 32
```
