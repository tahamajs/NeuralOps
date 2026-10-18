# Benchmark Run Summary: nasc_attention

- **Run ID**: `kd_nasc_attention_C_20261012_153046`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T15:40:40.590487+00:00
- **Git Commit**: `032c711cdf` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `nasc_attention` |
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
| **Peak VRAM** | 2835.99 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module nasc_attention --model gpt2 --dataset wikitext2 --rank 32
```
