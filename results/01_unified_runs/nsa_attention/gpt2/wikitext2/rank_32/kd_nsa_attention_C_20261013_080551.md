# Benchmark Run Summary: nsa_attention

- **Run ID**: `kd_nsa_attention_C_20261013_080551`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-13T08:55:07.071854+00:00
- **Git Commit**: `449aa3dccb` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `nsa_attention` |
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
| **Peak VRAM** | 3084.31 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module nsa_attention --model gpt2 --dataset wikitext2 --rank 32
```
