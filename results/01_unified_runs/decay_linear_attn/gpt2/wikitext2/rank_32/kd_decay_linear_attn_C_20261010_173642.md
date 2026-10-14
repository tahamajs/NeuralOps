# Benchmark Run Summary: decay_linear_attn

- **Run ID**: `kd_decay_linear_attn_C_20261010_173642`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-10T14:12:38.790966+00:00
- **Git Commit**: `c5c0c7b5ef` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `decay_linear_attn` |
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
| **Peak VRAM** | 3381.03 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module decay_linear_attn --model gpt2 --dataset wikitext2 --rank 32
```
