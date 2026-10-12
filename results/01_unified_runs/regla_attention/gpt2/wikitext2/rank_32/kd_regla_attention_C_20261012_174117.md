# Benchmark Run Summary: regla_attention

- **Run ID**: `kd_regla_attention_C_20261012_174117`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T17:43:22.564961+00:00
- **Git Commit**: `d5cdf1d2f7` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `regla_attention` |
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
| **Peak VRAM** | 1698.62 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module regla_attention --model gpt2 --dataset wikitext2 --rank 32
```
