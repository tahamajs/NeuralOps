# Benchmark Run Summary: block_transformer_hiba

- **Run ID**: `kd_block_transformer_hiba_C_20261012_222933`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T22:32:02.420905+00:00
- **Git Commit**: `02c7e8075e` (dirty=True)

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
| **Peak VRAM** | 5399.82 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module block_transformer_hiba --model gpt2 --dataset wikitext2 --rank 32
```
