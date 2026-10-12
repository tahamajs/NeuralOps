# Benchmark Run Summary: sink_attention

- **Run ID**: `kd_sink_attention_C_20261012_061448`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-12T06:17:30.659194+00:00
- **Git Commit**: `e744a85270` (dirty=True)

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `sink_attention` |
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
| **Peak VRAM** | 2836.77 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module sink_attention --model gpt2 --dataset wikitext2 --rank 32
```
