# Benchmark Run Summary: mixattention_moska

- **Run ID**: `kd_mixattention_moska_C_20261012_034444`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T03:47:14.080950+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `mixattention_moska` |
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
| **Peak VRAM** | 2871.52 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module mixattention_moska --model gpt2 --dataset wikitext2 --rank 32
```
