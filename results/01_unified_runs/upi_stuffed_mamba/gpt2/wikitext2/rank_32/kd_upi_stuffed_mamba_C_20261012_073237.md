# Benchmark Run Summary: upi_stuffed_mamba

- **Run ID**: `kd_upi_stuffed_mamba_C_20261012_073237`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T07:37:55.660998+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `upi_stuffed_mamba` |
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
| **Peak VRAM** | 2853.21 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module upi_stuffed_mamba --model gpt2 --dataset wikitext2 --rank 32
```
