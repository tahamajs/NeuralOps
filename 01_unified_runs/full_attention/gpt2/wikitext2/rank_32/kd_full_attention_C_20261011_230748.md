# Benchmark Run Summary: full_attention

- **Run ID**: `kd_full_attention_C_20261011_230748`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T23:11:30.360609+00:00
- **Git Commit**: `f9c917f1d0` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `full_attention` |
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
| **Peak VRAM** | 4602.83 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module full_attention --model gpt2 --dataset wikitext2 --rank 32
```
