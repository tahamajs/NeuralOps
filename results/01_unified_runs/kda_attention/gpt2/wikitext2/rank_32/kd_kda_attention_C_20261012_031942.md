# Benchmark Run Summary: kda_attention

- **Run ID**: `kd_kda_attention_C_20261012_031942`
- **Status**: **FAILED**
- **Timestamp**: 2026-10-12T03:21:59.419697+00:00
- **Git Commit**: `e5511f9a04` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `kda_attention` |
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
| **Peak VRAM** | 2873.57 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module kda_attention --model gpt2 --dataset wikitext2 --rank 32
```
