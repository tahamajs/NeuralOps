# Benchmark Run Summary: aa_svd_compressor

- **Run ID**: `kd_aa_svd_compressor_C_20261012_080506`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T08:07:31.122654+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `aa_svd_compressor` |
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
| **Peak VRAM** | 3015.04 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module aa_svd_compressor --model gpt2 --dataset wikitext2 --rank 32
```
