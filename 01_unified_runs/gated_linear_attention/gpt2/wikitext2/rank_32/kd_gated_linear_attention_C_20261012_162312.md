# Benchmark Run Summary: gated_linear_attention

- **Run ID**: `kd_gated_linear_attention_C_20261012_162312`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T16:24:36.351802+00:00
- **Git Commit**: `48699001d8` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `gated_linear_attention` |
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
| **Peak VRAM** | 1672.32 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module gated_linear_attention --model gpt2 --dataset wikitext2 --rank 32
```
