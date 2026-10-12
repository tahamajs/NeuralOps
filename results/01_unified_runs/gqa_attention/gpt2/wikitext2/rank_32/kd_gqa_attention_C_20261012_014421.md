# Benchmark Run Summary: gqa_attention

- **Run ID**: `kd_gqa_attention_C_20261012_014421`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T01:47:06.060652+00:00
- **Git Commit**: `842433a75e` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `gqa_attention` |
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
| **Peak VRAM** | 2813.47 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module gqa_attention --model gpt2 --dataset wikitext2 --rank 32
```
