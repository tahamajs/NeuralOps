# Benchmark Run Summary: flash_attention3_fp8

- **Run ID**: `kd_flash_attention3_fp8_C_20261011_230426`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T23:07:27.847588+00:00
- **Git Commit**: `c51c76b1c1` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `flash_attention3_fp8` |
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
python -m neuralops.benchmarks run --module flash_attention3_fp8 --model gpt2 --dataset wikitext2 --rank 32
```
