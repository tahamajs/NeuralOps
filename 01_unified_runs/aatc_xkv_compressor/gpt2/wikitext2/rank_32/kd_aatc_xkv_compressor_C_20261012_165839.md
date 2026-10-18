# Benchmark Run Summary: aatc_xkv_compressor

- **Run ID**: `kd_aatc_xkv_compressor_C_20261012_165839`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T17:07:19.515188+00:00
- **Git Commit**: `acc158890b` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `aatc_xkv_compressor` |
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
| **Peak VRAM** | 2044.95 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module aatc_xkv_compressor --model gpt2 --dataset wikitext2 --rank 32
```
