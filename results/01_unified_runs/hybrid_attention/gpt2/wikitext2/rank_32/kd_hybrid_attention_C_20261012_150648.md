# Benchmark Run Summary: hybrid_attention

- **Run ID**: `kd_hybrid_attention_C_20261012_150648`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T15:07:20.996978+00:00
- **Git Commit**: `38527665ca` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `hybrid_attention` |
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
| **Peak VRAM** | 3187.51 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module hybrid_attention --model gpt2 --dataset wikitext2 --rank 32
```
