# Benchmark Run Summary: cosformer_attention

- **Run ID**: `kd_cosformer_attention_C_20261011_222835`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T22:30:56.610716+00:00
- **Git Commit**: `671b506b86` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `cosformer_attention` |
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
| **Peak VRAM** | 4669.19 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module cosformer_attention --model gpt2 --dataset wikitext2 --rank 32
```
