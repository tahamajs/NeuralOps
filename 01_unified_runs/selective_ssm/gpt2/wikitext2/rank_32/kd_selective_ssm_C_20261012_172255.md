# Benchmark Run Summary: selective_ssm

- **Run ID**: `kd_selective_ssm_C_20261012_172255`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T17:30:07.630322+00:00
- **Git Commit**: `f7497ec808` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `selective_ssm` |
| **Model** | `gpt2` |
| **Dataset** | `wikitext2` |
| **Rank** | `32` |
| **Epochs** | `1` |
| **Batch Size** | `2` |

## Primary Metrics
| Metric | Value |
|---|---|
| **Accuracy** | N/A |
| **F1 Score** | N/A |
| **Loss** | N/A |
| **Duration (s)** | 0.0s |
| **Peak VRAM** | 3064.51 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module selective_ssm --model gpt2 --dataset wikitext2 --rank 32
```
