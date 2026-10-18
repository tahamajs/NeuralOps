# Benchmark Run Summary: cross_attention

- **Run ID**: `kd_cross_attention_C_20261011_223103`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T22:34:00.591487+00:00
- **Git Commit**: `671b506b86` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `cross_attention` |
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
| **Peak VRAM** | 4602.35 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module cross_attention --model gpt2 --dataset wikitext2 --rank 32
```
