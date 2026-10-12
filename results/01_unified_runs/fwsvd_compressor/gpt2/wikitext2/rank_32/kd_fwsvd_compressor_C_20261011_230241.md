# Benchmark Run Summary: fwsvd_compressor

- **Run ID**: `kd_fwsvd_compressor_C_20261011_230241`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T23:03:00.354426+00:00
- **Git Commit**: `3aeedeb48c` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `fwsvd_compressor` |
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
| **Peak VRAM** | 1108.06 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module fwsvd_compressor --model gpt2 --dataset wikitext2 --rank 32
```
