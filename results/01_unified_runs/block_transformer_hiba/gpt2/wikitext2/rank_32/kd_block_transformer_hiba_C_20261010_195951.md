# Benchmark Run Summary: block_transformer_hiba

- **Run ID**: `kd_block_transformer_hiba_C_20261010_195951`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-10T16:32:44.170871+00:00
- **Git Commit**: `6e09c09271` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `block_transformer_hiba` |
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
| **Peak VRAM** | 3413.84 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module block_transformer_hiba --model gpt2 --dataset wikitext2 --rank 32
```
