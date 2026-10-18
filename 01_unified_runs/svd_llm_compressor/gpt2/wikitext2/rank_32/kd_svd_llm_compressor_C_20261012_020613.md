# Benchmark Run Summary: svd_llm_compressor

- **Run ID**: `kd_svd_llm_compressor_C_20261012_020613`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T02:06:41.739211+00:00
- **Git Commit**: `e1bd7d4d1d` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `svd_llm_compressor` |
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
| **Peak VRAM** | 544.77 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module svd_llm_compressor --model gpt2 --dataset wikitext2 --rank 32
```
