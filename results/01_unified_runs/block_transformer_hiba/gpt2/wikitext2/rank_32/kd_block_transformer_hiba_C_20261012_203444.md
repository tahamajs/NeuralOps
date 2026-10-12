# Benchmark Run Summary: block_transformer_hiba

- **Run ID**: `kd_block_transformer_hiba_C_20261012_203444`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T20:36:02.917384+00:00
- **Git Commit**: `0d365d9f33` (dirty=True)

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
| **Peak VRAM** | 2469.93 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module block_transformer_hiba --model gpt2 --dataset wikitext2 --rank 32
```
