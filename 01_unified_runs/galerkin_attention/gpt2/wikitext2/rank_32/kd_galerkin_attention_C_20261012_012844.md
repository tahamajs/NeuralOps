# Benchmark Run Summary: galerkin_attention

- **Run ID**: `kd_galerkin_attention_C_20261012_012844`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T01:31:40.818480+00:00
- **Git Commit**: `9bdbef13ec` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `galerkin_attention` |
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
| **Peak VRAM** | 2841.65 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module galerkin_attention --model gpt2 --dataset wikitext2 --rank 32
```
