# Benchmark Run Summary: admm_compressor

- **Run ID**: `kd_admm_compressor_C_20261011_131045`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T09:43:27.939475+00:00
- **Git Commit**: `363d9d95c7` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `admm_compressor` |
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
| **Peak VRAM** | 2523.88 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module admm_compressor --model gpt2 --dataset wikitext2 --rank 32
```
