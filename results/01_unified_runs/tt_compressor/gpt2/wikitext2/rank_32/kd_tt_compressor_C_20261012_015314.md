# Benchmark Run Summary: tt_compressor

- **Run ID**: `kd_tt_compressor_C_20261012_015314`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T01:53:36.715002+00:00
- **Git Commit**: `37226964a0` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `tt_compressor` |
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
| **Peak VRAM** | 618.89 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module tt_compressor --model gpt2 --dataset wikitext2 --rank 32
```
