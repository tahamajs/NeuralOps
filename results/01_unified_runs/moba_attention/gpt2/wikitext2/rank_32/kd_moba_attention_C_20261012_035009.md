# Benchmark Run Summary: moba_attention

- **Run ID**: `kd_moba_attention_C_20261012_035009`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T03:53:22.634497+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `moba_attention` |
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
| **Peak VRAM** | 2868.3 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module moba_attention --model gpt2 --dataset wikitext2 --rank 32
```
