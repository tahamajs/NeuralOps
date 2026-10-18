# Benchmark Run Summary: factorised_matrix_state

- **Run ID**: `kd_factorised_matrix_state_C_20261011_225651`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T23:01:12.327114+00:00
- **Git Commit**: `a95c1af708` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `factorised_matrix_state` |
| **Model** | `gpt2` |
| **Dataset** | `wikitext2` |
| **Rank** | `32` |
| **Epochs** | `1` |
| **Batch Size** | `12` |

## Primary Metrics
| Metric | Value |
|---|---|
| **Accuracy** | N/A |
| **F1 Score** | N/A |
| **Loss** | N/A |
| **Duration (s)** | 0.0s |
| **Peak VRAM** | 4603.54 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module factorised_matrix_state --model gpt2 --dataset wikitext2 --rank 32
```
