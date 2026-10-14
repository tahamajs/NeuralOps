# Benchmark Run Summary: sink_attention

- **Run ID**: `kd_sink_attention_C_20261010_202031`
- **Status**: **SUCCESS**
- **Timestamp**: 2026-10-10T16:53:31.235947+00:00
- **Git Commit**: `6e09c09271` (dirty=True)

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
| **Peak VRAM** | 3377.71 MB |

## Reproduction Command
```bash
python -m neuralops.benchmarks run --module sink_attention --model gpt2 --dataset wikitext2 --rank 32
```
