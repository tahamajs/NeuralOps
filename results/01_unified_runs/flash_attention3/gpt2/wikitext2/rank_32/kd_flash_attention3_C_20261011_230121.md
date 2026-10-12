# Benchmark Run Summary: flash_attention3

- **Run ID**: `kd_flash_attention3_C_20261011_230121`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T23:04:19.746135+00:00
- **Git Commit**: `3aeedeb48c` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `flash_attention3` |
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
| **Peak VRAM** | 4601.4 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module flash_attention3 --model gpt2 --dataset wikitext2 --rank 32
```
