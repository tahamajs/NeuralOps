# Benchmark Run Summary: primal_attention

- **Run ID**: `kd_primal_attention_C_20261012_044116`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T04:44:57.659350+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `primal_attention` |
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
| **Peak VRAM** | 2839.02 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module primal_attention --model gpt2 --dataset wikitext2 --rank 32
```
