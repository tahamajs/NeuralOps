# Benchmark Run Summary: triton_cosformer

- **Run ID**: `kd_triton_cosformer_C_20261012_072226`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T07:25:25.639427+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `triton_cosformer` |
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
| **Peak VRAM** | 3090.62 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module triton_cosformer --model gpt2 --dataset wikitext2 --rank 32
```
