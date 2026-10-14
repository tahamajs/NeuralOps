# Benchmark Run Summary: lola_attention

- **Run ID**: `kd_lola_attention_C_20261010_201048`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-10T16:45:24.632868+00:00
- **Git Commit**: `6e09c09271` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `lola_attention` |
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
| **Peak VRAM** | 3310.73 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module lola_attention --model gpt2 --dataset wikitext2 --rank 32
```
