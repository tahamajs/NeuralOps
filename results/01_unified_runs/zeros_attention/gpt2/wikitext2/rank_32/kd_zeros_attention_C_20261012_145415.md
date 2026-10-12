# Benchmark Run Summary: zeros_attention

- **Run ID**: `kd_zeros_attention_C_20261012_145415`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T14:54:42.564003+00:00
- **Git Commit**: `38527665ca` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `zeros_attention` |
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
| **Peak VRAM** | 2837.34 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module zeros_attention --model gpt2 --dataset wikitext2 --rank 32
```
