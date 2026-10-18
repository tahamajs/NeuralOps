# Benchmark Run Summary: gla_attention

- **Run ID**: `kd_gla_attention_C_20261013_061527`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-13T06:19:21.065281+00:00
- **Git Commit**: `127c5e44eb` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `gla_attention` |
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
| **Peak VRAM** | 3807.35 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module gla_attention --model gpt2 --dataset wikitext2 --rank 32
```
