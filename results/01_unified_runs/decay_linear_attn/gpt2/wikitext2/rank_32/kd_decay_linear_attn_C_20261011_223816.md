# Benchmark Run Summary: decay_linear_attn

- **Run ID**: `kd_decay_linear_attn_C_20261011_223816`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T22:42:13.157489+00:00
- **Git Commit**: `afe6613ee9` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `decay_linear_attn` |
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
| **Peak VRAM** | 4603.69 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module decay_linear_attn --model gpt2 --dataset wikitext2 --rank 32
```
