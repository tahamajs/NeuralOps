# Benchmark Run Summary: vig_attention

- **Run ID**: `kd_vig_attention_C_20261012_073810`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T07:46:24.479192+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `vig_attention` |
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
| **Peak VRAM** | 2875.38 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module vig_attention --model gpt2 --dataset wikitext2 --rank 32
```
