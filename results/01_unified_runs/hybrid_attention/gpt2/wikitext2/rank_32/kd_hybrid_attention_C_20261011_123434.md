# Benchmark Run Summary: hybrid_attention

- **Run ID**: `kd_hybrid_attention_C_20261011_123434`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T09:05:59.599192+00:00
- **Git Commit**: `3476950323` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `hybrid_attention` |
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
| **Peak VRAM** | 1189.35 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module hybrid_attention --model gpt2 --dataset wikitext2 --rank 32
```
