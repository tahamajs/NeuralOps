# Benchmark Run Summary: hybridkv_compressor

- **Run ID**: `kd_hybridkv_compressor_C_20261012_170830`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T17:09:12.645671+00:00
- **Git Commit**: `acc158890b` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `hybridkv_compressor` |
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
| **Peak VRAM** | 2029.77 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module hybridkv_compressor --model gpt2 --dataset wikitext2 --rank 32
```
