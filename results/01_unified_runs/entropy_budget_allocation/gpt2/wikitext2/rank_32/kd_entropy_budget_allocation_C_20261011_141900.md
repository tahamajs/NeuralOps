# Benchmark Run Summary: entropy_budget_allocation

- **Run ID**: `kd_entropy_budget_allocation_C_20261011_141900`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T10:52:24.066225+00:00
- **Git Commit**: `0f67bb39bf` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `entropy_budget_allocation` |
| **Model** | `gpt2` |
| **Dataset** | `wikitext2` |
| **Rank** | `32` |
| **Epochs** | `1` |
| **Batch Size** | `4` |

## Primary Metrics
| Metric | Value |
|---|---|
| **Accuracy** | N/A |
| **F1 Score** | N/A |
| **Loss** | N/A |
| **Duration (s)** | 0.0s |
| **Peak VRAM** | 2049.83 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module entropy_budget_allocation --model gpt2 --dataset wikitext2 --rank 32
```
