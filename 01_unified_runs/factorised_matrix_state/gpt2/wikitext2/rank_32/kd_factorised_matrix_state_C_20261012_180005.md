# Benchmark Run Summary: factorised_matrix_state

- **Run ID**: `kd_factorised_matrix_state_C_20261012_180005`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T18:01:45.153560+00:00
- **Git Commit**: `4b230f0d85` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `factorised_matrix_state` |
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
| **Peak VRAM** | 2837.41 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module factorised_matrix_state --model gpt2 --dataset wikitext2 --rank 32
```
