# Benchmark Run Summary: fwsvd_compressor

- **Run ID**: `kd_fwsvd_compressor_C_20261012_015600`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T01:56:19.729403+00:00
- **Git Commit**: `20116686f5` (dirty=True)

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
| **Peak VRAM** | 618.97 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module fwsvd_compressor --model gpt2 --dataset wikitext2 --rank 32
```
