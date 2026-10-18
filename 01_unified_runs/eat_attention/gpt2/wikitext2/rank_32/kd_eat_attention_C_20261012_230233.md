# Benchmark Run Summary: eat_attention

- **Run ID**: `kd_eat_attention_C_20261012_230233`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T23:03:03.074716+00:00
- **Git Commit**: `ecd28978a5` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `eat_attention` |
| **Model** | `gpt2` |
| **Dataset** | `wikitext2` |
| **Rank** | `32` |
| **Epochs** | `1` |
| **Batch Size** | `4` |

## Primary Metrics
| Metric | Value |
|---|---|
| **Accuracy** | N/A |
| **F1 Score** | N/A |
| **Loss** | N/A |
| **Duration (s)** | 0.0s |
| **Peak VRAM** | 3325.47 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module eat_attention --model gpt2 --dataset wikitext2 --rank 32
```
