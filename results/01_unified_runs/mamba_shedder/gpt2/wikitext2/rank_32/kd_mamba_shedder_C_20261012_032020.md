# Benchmark Run Summary: mamba_shedder

- **Run ID**: `kd_mamba_shedder_C_20261012_032020`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T03:24:26.926969+00:00
- **Git Commit**: `e5511f9a04` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `mamba_shedder` |
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
| **Peak VRAM** | 3007.86 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module mamba_shedder --model gpt2 --dataset wikitext2 --rank 32
```
