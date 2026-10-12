# Benchmark Run Summary: performer_attention

- **Run ID**: `kd_performer_attention_C_20261012_145700`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T14:57:31.861123+00:00
- **Git Commit**: `38527665ca` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `performer_attention` |
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
| **Peak VRAM** | 2874.52 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module performer_attention --model gpt2 --dataset wikitext2 --rank 32
```
