# Benchmark Run Summary: regla_attention

- **Run ID**: `kd_regla_attention_C_20261010_042753`
- **Status**: **FAILED**
- **Timestamp**: 2026-10-10T01:07:28.514886+00:00
- **Git Commit**: `a68a7e0b4e` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `regla_attention` |
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
| **Peak VRAM** | 3426.29 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module regla_attention --model gpt2 --dataset wikitext2 --rank 32
```
