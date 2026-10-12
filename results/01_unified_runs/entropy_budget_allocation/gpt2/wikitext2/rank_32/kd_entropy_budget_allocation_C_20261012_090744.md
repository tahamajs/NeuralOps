# Benchmark Run Summary: entropy_budget_allocation

- **Run ID**: `kd_entropy_budget_allocation_C_20261012_090744`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T09:10:03.822945+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `entropy_budget_allocation` |
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
| **Peak VRAM** | 2677.11 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module entropy_budget_allocation --model gpt2 --dataset wikitext2 --rank 32
```
