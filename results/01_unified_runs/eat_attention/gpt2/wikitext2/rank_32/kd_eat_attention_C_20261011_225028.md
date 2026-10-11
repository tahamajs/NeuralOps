# Benchmark Run Summary: eat_attention

- **Run ID**: `kd_eat_attention_C_20261011_225028`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T22:52:56.432838+00:00
- **Git Commit**: `55862cba9b` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `eat_attention` |
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
| **Peak VRAM** | 4668.24 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module eat_attention --model gpt2 --dataset wikitext2 --rank 32
```
