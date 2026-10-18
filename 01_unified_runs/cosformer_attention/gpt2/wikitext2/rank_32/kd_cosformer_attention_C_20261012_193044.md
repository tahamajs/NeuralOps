# Benchmark Run Summary: cosformer_attention

- **Run ID**: `kd_cosformer_attention_C_20261012_193044`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T19:32:23.856608+00:00
- **Git Commit**: `450495675d` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `cosformer_attention` |
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
| **Peak VRAM** | 2884.56 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module cosformer_attention --model gpt2 --dataset wikitext2 --rank 32
```
