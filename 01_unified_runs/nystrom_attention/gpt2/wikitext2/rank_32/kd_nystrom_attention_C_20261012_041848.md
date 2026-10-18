# Benchmark Run Summary: nystrom_attention

- **Run ID**: `kd_nystrom_attention_C_20261012_041848`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T04:21:35.150464+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `nystrom_attention` |
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
| **Peak VRAM** | 2840.77 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module nystrom_attention --model gpt2 --dataset wikitext2 --rank 32
```
