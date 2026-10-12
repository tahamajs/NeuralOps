# Benchmark Run Summary: block_transformer_hiba

- **Run ID**: `kd_block_transformer_hiba_C_20261012_222648`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T22:29:10.660587+00:00
- **Git Commit**: `d4f15e70d3` (dirty=True)

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
| **Peak VRAM** | 5399.57 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module block_transformer_hiba --model gpt2 --dataset wikitext2 --rank 32
```
