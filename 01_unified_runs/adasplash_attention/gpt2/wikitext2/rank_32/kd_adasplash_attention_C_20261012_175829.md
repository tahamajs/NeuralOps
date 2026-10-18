# Benchmark Run Summary: adasplash_attention

- **Run ID**: `kd_adasplash_attention_C_20261012_175829`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T17:59:53.057074+00:00
- **Git Commit**: `4b230f0d85` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `adasplash_attention` |
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
| **Peak VRAM** | 2838.27 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module adasplash_attention --model gpt2 --dataset wikitext2 --rank 32
```
