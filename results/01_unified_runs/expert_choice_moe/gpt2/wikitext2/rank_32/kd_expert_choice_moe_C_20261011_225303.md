# Benchmark Run Summary: expert_choice_moe

- **Run ID**: `kd_expert_choice_moe_C_20261011_225303`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-11T22:56:45.072986+00:00
- **Git Commit**: `0d2955ce48` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `expert_choice_moe` |
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
| **Peak VRAM** | 4664.15 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module expert_choice_moe --model gpt2 --dataset wikitext2 --rank 32
```
