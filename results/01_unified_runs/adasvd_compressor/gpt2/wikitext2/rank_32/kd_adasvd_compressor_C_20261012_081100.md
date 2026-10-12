# Benchmark Run Summary: adasvd_compressor

- **Run ID**: `kd_adasvd_compressor_C_20261012_081100`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T08:13:33.610689+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `adasvd_compressor` |
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
| **Peak VRAM** | 3045.55 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module adasvd_compressor --model gpt2 --dataset wikitext2 --rank 32
```
