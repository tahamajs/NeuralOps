# Benchmark Run Summary: regla_attention

- **Run ID**: `kd_regla_attention_C_20261012_051514`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T05:25:04.620907+00:00
- **Git Commit**: `e744a85270` (dirty=True)

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
| **Peak VRAM** | 2885.28 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module regla_attention --model gpt2 --dataset wikitext2 --rank 32
```
