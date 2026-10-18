# Benchmark Run Summary: zeros_attention

- **Run ID**: `kd_zeros_attention_C_20261012_175401`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T17:56:14.936335+00:00
- **Git Commit**: `4b230f0d85` (dirty=True)

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
| **Peak VRAM** | 2838.69 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module zeros_attention --model gpt2 --dataset wikitext2 --rank 32
```
