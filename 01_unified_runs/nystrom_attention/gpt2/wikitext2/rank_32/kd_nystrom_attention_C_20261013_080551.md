# Benchmark Run Summary: nystrom_attention

- **Run ID**: `kd_nystrom_attention_C_20261013_080551`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-13T08:44:47.100481+00:00
- **Git Commit**: `55485ae036` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `nystrom_attention` |
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
| **Peak VRAM** | 3022.07 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module nystrom_attention --model gpt2 --dataset wikitext2 --rank 32
```
