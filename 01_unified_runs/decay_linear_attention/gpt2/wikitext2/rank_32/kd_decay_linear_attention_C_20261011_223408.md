# Benchmark Run Summary: decay_linear_attention

- **Run ID**: `kd_decay_linear_attention_C_20261011_223408`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T22:38:09.087956+00:00
- **Git Commit**: `afe6613ee9` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `decay_linear_attention` |
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
| **Peak VRAM** | 4604.27 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module decay_linear_attention --model gpt2 --dataset wikitext2 --rank 32
```
