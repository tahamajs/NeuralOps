# Benchmark Run Summary: coopq_quantizer

- **Run ID**: `kd_coopq_quantizer_C_20261012_084014`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T08:43:17.412802+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `coopq_quantizer` |
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
| **Peak VRAM** | 3083.15 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module coopq_quantizer --model gpt2 --dataset wikitext2 --rank 32
```
