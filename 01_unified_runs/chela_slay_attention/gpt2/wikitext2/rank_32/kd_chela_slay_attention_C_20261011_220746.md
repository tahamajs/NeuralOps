# Benchmark Run Summary: chela_slay_attention

- **Run ID**: `kd_chela_slay_attention_C_20261011_220746`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T22:14:37.078853+00:00
- **Git Commit**: `671b506b86` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `chela_slay_attention` |
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
python -m neuralops.benchmarks run --module chela_slay_attention --model gpt2 --dataset wikitext2 --rank 32
```
